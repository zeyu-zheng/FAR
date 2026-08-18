"""External agent backend for the Solve, Judge, and Grade stages.

These three stages do not call the model API directly. Each runs an `opencode`
agent in a per-candidate working directory that holds the stage's inputs as
files, exactly as the user prompts instruct the agent to read them.

The workspace layout is the contract between the three stages:

    <work_root>/<work_name>/
        input.json      written by write_input_json()  -- read by all three
        solution.md     written by Solve               -- read by Judge, Grade
        judge_NN.md     written by Judge (one per judge run)
        judge.md        written by Judge (concatenated) -- read by Grade
        grade.md        written by Grade

Because Judge and Grade can run either straight after Solve or as separate
passes over an earlier run's output, `write_input_json` is the single writer
for input.json, so that a stage run over an earlier sweep hands its agent the
same bytes as one that ran inline.
"""

import json
import os
import pty
import select
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from src.prompts import (
    GRADER_AGENT_FRONTMATTER,
    GRADER_SYSTEM_PROMPT,
    JUDGE_AGENT_FRONTMATTER,
    JUDGE_SYSTEM_PROMPT,
    PROVER_AGENT_FRONTMATTER,
    PROVER_SYSTEM_PROMPT,
)
from src.utils import PipelineCancelled, check_cancelled, wait_retry

RETRY_SLEEP = 60

PROVER_AGENT = "prover"
JUDGE_AGENT = "judge"
GRADER_AGENT = "grader"

# First-line tokens each agent is required to emit.
SOURCE_WORDS = {"KNOWN", "NEW", "FIX", "NONE"}
JUDGE_WORDS = {"PASS", "FAIL", "KNOWN"}
QUALITY_WORDS = {"KNOWN", "TYPE1", "TYPE2", "TYPE3"}
QUALITY_LABELS = {"KNOWN": "known", "TYPE1": "type1", "TYPE2": "type2", "TYPE3": "type3"}


# ── Workspace ───────────────────────────────────────────────────────────────

# Keys of a task record that input.json is built from, plus the two that
# identify it. Any stage that persists a record which a later stage may re-open
# must carry all of these.
TASK_KEYS = (
    "row_index",
    "candidate_index",
    "title",
    "authors",
    "conjecture_label",
    "conjecture_section",
    "conjecture_text",
    "sources",
)


def write_input_json(task: dict[str, Any], body: str, work_dir: Path) -> Path:
    """Write the agent-visible input.json.

    Only what an agent can act on. Check's `importance` and `difficulty` are
    left out: the paper records them for the effort-allocation analysis, and
    handing a prover a difficulty score works against the instruction not to
    stop just because a statement is labelled open. Its `reason` is left out
    too -- every candidate that reaches Solve is open, so the sentence adds
    nothing the `sources` do not carry. `candidate_index` is a pipeline
    identifier, not something the agent can use.
    """
    path = work_dir / "input.json"
    path.write_text(
        json.dumps(
            {
                "title": task["title"],
                "authors": task["authors"],
                "text": body,
                "sources": task.get("sources") or [],
                "conjecture": {
                    "label": task.get("conjecture_label"),
                    "section": task.get("conjecture_section"),
                    "text": task["conjecture_text"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def task_from_record(item: dict[str, Any]) -> dict[str, Any]:
    """Recover the task fields from a persisted stage record.

    Lets Judge and Grade rebuild a workspace identical to the one Solve used,
    without re-running the earlier stage.
    """
    return {key: item.get(key) for key in TASK_KEYS}


def work_name_for(task: dict[str, Any]) -> str:
    row, candidate = result_key(task)
    return f"row_{row}_candidate_{candidate}"


def restore_workspace(item: dict[str, Any], body: str, work_root: Path) -> Path:
    """Rebuild an agent workspace from a persisted record.

    Writes back what the earlier stages produced, so a stage run over an older
    sweep hands its agent the same files as one that ran inline. Only a record
    that has been judged carries a judgement, which is what decides whether
    judge.md is there for the grader to read.
    """
    work_dir = prepare_work_dir(task_from_record(item), body, work_root)
    (work_dir / "solution.md").write_text(
        str(item.get("solution") or "").strip() or "(no solution)", encoding="utf-8"
    )
    if "judgement" in item:
        (work_dir / "judge.md").write_text(
            str(item.get("judgement") or "").strip() or "(no judge output)", encoding="utf-8"
        )
    return work_dir


def prepare_work_dir(task: dict[str, Any], body: str, work_root: Path) -> Path:
    work_dir = (work_root / work_name_for(task)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_input_json(task, body, work_dir)
    return work_dir


# ── opencode backend ────────────────────────────────────────────────────────


def require_opencode() -> None:
    if shutil.which("opencode") is None:
        raise SystemExit(
            "opencode executable not found on PATH; install opencode or update PATH "
            "before running the solve, judge, or grade stages."
        )


def install_opencode_agents() -> None:
    agent_dir = Path(".opencode/agents")
    agent_dir.mkdir(parents=True, exist_ok=True)
    for name, frontmatter, system_prompt in (
        (PROVER_AGENT, PROVER_AGENT_FRONTMATTER, PROVER_SYSTEM_PROMPT),
        (JUDGE_AGENT, JUDGE_AGENT_FRONTMATTER, JUDGE_SYSTEM_PROMPT),
        (GRADER_AGENT, GRADER_AGENT_FRONTMATTER, GRADER_SYSTEM_PROMPT),
    ):
        definition = f"{frontmatter.strip()}\n\n{system_prompt.strip()}\n"
        (agent_dir / f"{name}.md").write_text(definition, encoding="utf-8")


def opencode_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def run_opencode_command(command: list[str], stop_event: threading.Event | None) -> tuple[int, str]:
    """Run opencode on a pty and stream its output, honouring cancellation."""
    master, slave = pty.openpty()
    output = bytearray()
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    cancelled = False
    try:
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break
            ready, _, _ = select.select([master], [], [], 1)
            if not ready:
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
        if not cancelled:
            while True:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if cancelled:
        raise PipelineCancelled("pipeline cancelled")
    return proc.wait(), output.decode(errors="replace")


def run_agent(
    model: str,
    effort: str | None,
    agent: str,
    message: str,
    work_dir: Path,
    retries: int,
    expected_first_words: set[str],
    attachments: list[Path],
    stop_event: threading.Event | None = None,
) -> str:
    """Run one agent turn and return its validated final text."""
    work_dir = work_dir.resolve()
    last_error = None
    for attempt in range(1, retries + 2):
        check_cancelled(stop_event)
        try:
            command = [
                "opencode",
                "run",
                "--format",
                "json",
                "--model",
                opencode_model(model),
                "--dir",
                str(work_dir),
                "--agent",
                agent,
            ]
            if effort:
                command.extend(["--variant", effort])
            command.append(message)
            for path in attachments:
                command.extend(["--file", str(path.resolve())])
            returncode, output = run_opencode_command(command, stop_event)
            if returncode != 0:
                raise RuntimeError(output.strip())
            parts = []
            for line in output.splitlines():
                try:
                    event = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                part = event.get("part") or {}
                if event.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
            text = select_expected_output(parts, expected_first_words)
            if not text:
                raise RuntimeError("empty opencode output")
            if parse_first_word(text, expected_first_words, "") == "":
                raise RuntimeError(f"unexpected first word in opencode output: {text.splitlines()[0][:80]}")
            return text
        except PipelineCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - retry transient opencode/backend failures
            last_error = exc
            if attempt <= retries:
                wait_retry(RETRY_SLEEP, stop_event)
    raise RuntimeError(str(last_error))


def parse_first_word(text: str, allowed: set[str], default: str) -> str:
    word = (text.strip().split() or [default])[0].upper()
    return word if word in allowed else default


def select_expected_output(text_parts: list[str], expected_first_words: set[str]) -> str:
    for text in reversed(text_parts):
        stripped = text.strip()
        if parse_first_word(stripped, expected_first_words, ""):
            return stripped
        lines = stripped.splitlines()
        for index, line in enumerate(lines):
            if parse_first_word(line, expected_first_words, ""):
                return "\n".join(lines[index:]).strip()
    return "\n".join(text_parts).strip()


# ── Outcome classification ──────────────────────────────────────────────────


def aggregate_judge_verdict(verdicts: list[str]) -> str:
    """
    PASS a solution if and only if all judges pass.
    KNOWN a solution if and only if at least one judge returns KNOWN and no judge returns FAIL.
    FAIL a solution if and only if at least one judge returns FAIL.
    """
    if not verdicts:
        return "SKIP"
    if "FAIL" in verdicts:
        return "FAIL"
    if "KNOWN" in verdicts:
        return "KNOWN"
    if all(verdict == "PASS" for verdict in verdicts):
        return "PASS"
    return "FAIL"


def classify_result(source: str, verdict: str) -> str:
    if source == "KNOWN" and verdict in {"PASS", "KNOWN"}:
        return "known"
    if source == "NEW" and verdict == "PASS":
        return "new"
    if source == "NEW" and verdict == "KNOWN":  # demotion from NEW to KNOWN
        return "known"
    if source == "FIX":
        return "fix"
    return "none"


def result_key(item: dict[str, Any]) -> tuple[int, int]:
    """Identity of one attempt.

    Keyed on positions this pipeline assigned -- the row's place in the corpus
    and the candidate's place in the paper -- rather than on `title`,
    which Extract's model reads off the paper and may word differently between
    runs.
    """
    return (int(item["row_index"]), int(item["candidate_index"]))


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
