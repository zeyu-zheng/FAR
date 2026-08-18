"""Shared utilities: LLM client, JSON parsing, JSONL IO, cancellation.

The LLM client backs the three Find stages (Label, Extract, Check). The three
later stages (Solve, Judge, Grade) run through an external agent CLI instead;
see agent.py.
"""

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import AzureOpenAI, OpenAI

# ── Environment ─────────────────────────────────────────────────────────────

# Every response the Find stages get, in one file next to the agent workspaces.
# It grows with the run and is never read back -- it is there for looking at a
# result afterwards and asking what the model actually said.
MODEL_OUTPUT_LOG = ".far/model_outputs.jsonl"

RESPONSES_API = "responses"
CHAT_API = "chat"

# Each Find stage has its own key and endpoint, so the three can sit behind
# three different providers.
STAGE_ENV = {
    "label": ("LABEL_API_KEY", "LABEL_BASE_URL"),
    "extract": ("EXTRACT_API_KEY", "EXTRACT_BASE_URL"),
    "check": ("CHECK_API_KEY", "CHECK_BASE_URL"),
}


def require_env(name: str) -> str:
    """Read a required credential, or explain how to set it."""
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(
        f"{name} is not set. Export it, or put it in the env file (default ~/.env):\n"
        f"    export {name}=...\n"
        "Each Find stage takes its own key and endpoint. "
        "See the Setup section of README.md."
    )


def load_env_file(path: str | None = "~/.env") -> None:
    """Load KEY=VALUE lines from an env file without overriding the environment."""
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ── Cancellation ────────────────────────────────────────────────────────────


class PipelineCancelled(RuntimeError):
    """Raised when a stop event fires, so a worker unwinds instead of retrying."""


def wait_retry(delay: float, stop_event: threading.Event | None) -> None:
    if stop_event is None:
        time.sleep(delay)
        return
    if stop_event.wait(delay):
        raise PipelineCancelled("pipeline cancelled")


def check_cancelled(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise PipelineCancelled("pipeline cancelled")


# ── Request throttle ────────────────────────────────────────────────────────

_REQUEST_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def throttle_requests(min_interval: float) -> None:
    if min_interval <= 0:
        return
    global _LAST_REQUEST_AT
    with _REQUEST_THROTTLE_LOCK:
        elapsed = time.time() - _LAST_REQUEST_AT
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _LAST_REQUEST_AT = time.time()


# ── Model output log ────────────────────────────────────────────────────────

_MODEL_OUTPUT_LOG_LOCK = threading.Lock()


def log_model_output(client: "LLMClient", content: str, metadata: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "web_search": client.web_search,
        "content": content,
        **metadata,
    }
    path = Path(MODEL_OUTPUT_LOG)
    with _MODEL_OUTPUT_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── LLM client ──────────────────────────────────────────────────────────────

# Transport-level retries. A long run against a rate-limited endpoint spends
# most of its failures here -- 429s, dropped connections, empty responses --
# and none of them are worth losing a paper over, so the ceiling is high and
# the gap is fixed. The wait is cancellable, so a stopping run does not sit
# through it.
HTTP_RETRIES = 128
HTTP_RETRY_DELAY = 10.0


@dataclass
class LLMClient:
    """One configured model endpoint, belonging to one Find stage.

    Supports both the Responses API and Chat Completions. The stage names the
    credentials: each one reads its own key and endpoint from the environment,
    so the three stages can sit behind three different providers.
    """

    stage: str
    model: str
    api_type: str = CHAT_API
    reasoning_effort: str | None = None
    web_search: bool = False
    timeout: float = 3600.0
    max_tokens: int = 32000
    request_min_interval: float = 0.0
    api_version: str = "2024-03-01-preview"
    http_retries: int = HTTP_RETRIES
    model_log: bool = True
    stop_event: threading.Event | None = None
    calls: int = 0
    client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        key_env, url_env = STAGE_ENV[self.stage]
        base_url = require_env(url_env)
        api_key = require_env(key_env)
        if "api.openai.com" in base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
        else:
            self.client = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=self.api_version,
                timeout=self.timeout,
            )

    async def complete(
        self,
        messages: list[dict[str, str]],
        validate_result: Callable[[str], dict[str, Any]] | None = None,
        max_validation_retries: int = 2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """One completion, retried with feedback while `validate_result` rejects it.

        A rejected answer goes back to the model with the reason, which is how a
        response that does not fit the schema gets a second chance without
        costing the whole paper. Web search, when the stage uses it, is only
        offered on the first attempt.
        """
        conversation: list[dict[str, Any]] = list(messages)
        attempts = 0

        while True:
            response = await asyncio.to_thread(
                self.send, conversation, self.web_search and attempts == 0, response_format
            )
            content = self.text_of(response)
            feedback = validate_result(content) if validate_result else None
            accepted = feedback is None or feedback.get("success", True)
            if self.model_log:
                log_model_output(
                    self,
                    content,
                    {"api_type": self.api_type, "attempt": attempts, "accepted": accepted, "call_count": self.calls},
                )
            if accepted:
                return content
            attempts += 1
            if attempts > max_validation_retries:
                return content
            if content:
                conversation.append({"role": "assistant", "content": content})
            conversation.append({"role": "user", "content": feedback["message"]})

    def send(self, conversation: list[dict[str, Any]], use_web_search: bool, response_format) -> Any:
        if self.api_type == RESPONSES_API:
            return self.create_responses_response(conversation, use_web_search)

        response = self.create_chat_response(conversation, use_web_search, response_format)
        if use_web_search:
            # A Check verdict reached without searching is worthless, so an
            # unused tool is treated as a failed request. `usage.extra` is not
            # an OpenAI field -- like the google_search tool below, it is a
            # convention of the endpoint this was built against.
            usage = json.loads(response.usage.model_dump_json()) if response.usage else {}
            searched = usage.get("extra", {}).get("web_search", 0)
            reasoned = usage.get("reasoning_tokens", 0)
            if not searched or not reasoned:
                raise RuntimeError(
                    f"web_search={searched}, reasoning_tokens={reasoned}: "
                    "model did not use web search or reasoning"
                )
        return response

    def text_of(self, response: Any) -> str:
        if self.api_type == RESPONSES_API:
            return extract_responses_text(response)
        return response.choices[0].message.content or ""

    def create_responses_response(self, conversation: list[dict[str, Any]], use_web_search: bool) -> Any:
        self.calls += 1
        params: dict[str, Any] = {"model": self.model, "input": conversation}
        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}
        if use_web_search:
            params["tools"] = [{"type": "web_search"}]
        return self.request(lambda: self.client.responses.create(**params))

    def create_chat_response(
        self,
        messages: list[dict[str, Any]],
        use_web_search: bool,
        response_format: dict[str, Any] | None,
    ) -> Any:
        self.calls += 1
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            params["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        if use_web_search:
            # Gemini's own OpenAI-compatible endpoint takes grounding through
            # extra_body instead. This shape is what the endpoint FAR was built
            # against expects; against Google directly it is ignored, and the
            # usage check above then rejects the response.
            params["tools"] = [{"type": "google_search"}]
            params["tool_choice"] = "auto"
        if response_format is not None:
            params["response_format"] = response_format
        return self.request(lambda: self.client.chat.completions.create(**params))

    # -- Transport -----------------------------------------------------------

    def request(self, send: Callable[[], Any]) -> Any:
        """Send one request, retrying transport failures until they stop."""
        last_error: Exception | None = None
        for attempt in range(self.http_retries + 1):
            check_cancelled(self.stop_event)
            throttle_requests(self.request_min_interval)
            try:
                return send()
            except PipelineCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - retry transport failures
                last_error = exc
                if attempt < self.http_retries:
                    wait_retry(HTTP_RETRY_DELAY, self.stop_event)
        raise RuntimeError(
            f"request failed after {self.http_retries + 1} attempts: {last_error}"
        ) from last_error


def extract_responses_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts)


# ── JSON helpers ────────────────────────────────────────────────────────────


def extract_first_json(text: str, required_keys: set[str] = set()) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(obj, dict) and required_keys.issubset(obj):
            return obj
    return None


def validate_score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return score


# How much of a paper each Find stage reads, in characters. Label only has to
# place the paper in a research direction, which the opening pages settle, and
# it reads the entire corpus, so it stays small. Extract has to find statements
# that cluster in the closing sections, so it reads far more -- and reads the
# end rather than the beginning, so that an unusually long paper loses its
# introduction rather than its open problems.
LABEL_MAX_CHARS = 64_000
EXTRACT_MAX_CHARS = 256_000


PAPER_FIELD = "text"


def paper_body(row: dict[str, Any]) -> str:
    if PAPER_FIELD not in row:
        raise ValueError(
            f"corpus row is missing the {PAPER_FIELD!r} field; "
            f"available fields: {sorted(row)[:8]}"
        )
    return str(row[PAPER_FIELD] or "")


def head(body: str) -> str:
    """The opening of the paper, for Label."""
    return body[:LABEL_MAX_CHARS]


def tail(body: str) -> str:
    """The closing of the paper, for Extract and for what the agents read."""
    return body[-EXTRACT_MAX_CHARS:]


# ── JSONL IO ────────────────────────────────────────────────────────────────

_APPEND_LOCK = threading.Lock()


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with _APPEND_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")

