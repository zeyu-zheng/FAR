"""External agent backend for the Solve, Judge, and Grade stages.

These three stages do not call the model API directly. Each runs an `opencode`
agent, locally or in Docker (see runtime.py), in a fresh workspace that holds
the stage's inputs as files, exactly as the user prompts instruct the agent to
read them.

The layout under each candidate's directory:

    <work_root>/row_R_candidate_C/
        input.json, solution.md, judge.md, grade.md   the candidate's records
        solve/  judge_001/ ...  grade/                one directory per stage or judge round
            workspace/                                rebuilt from the records for every attempt
            solution.md | judge.md | grade.md         the answer, saved by Python
            output_NNN.log                            opencode's raw output for attempt NNN
            session_NNN.json                          opencode's exported session for attempt NNN

NNN counts attempts at that one call -- retries, and re-runs over the same
directory -- so no attempt's log or session overwrites another's.

Because Judge and Grade can run either straight after Solve or as separate
passes over an earlier run's output, `write_input_json` is the single writer
for input.json, so that a stage run over an earlier sweep hands its agent the
same bytes as one that ran inline.
"""

import collections
import json
import os
import select
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from src.runtime import host_log, save_session
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


class OpencodeOutput:
    """Reads opencode's JSON event stream as it arrives.

    Keeps only what the pipeline needs -- the text parts, the session id, and a
    short tail of anything else for error messages; the full stream is in the
    attempt's output log.
    """

    def __init__(self) -> None:
        self.buffer = b""
        self.texts: list[str] = []
        self.session_id: str | None = None
        self.tail: collections.deque[str] = collections.deque(maxlen=20)

    def feed(self, chunk: bytes) -> None:
        *lines, self.buffer = (self.buffer + chunk).split(b"\n")
        for line in lines:
            self.line(line)

    def close(self) -> None:
        if self.buffer:
            self.line(self.buffer)
            self.buffer = b""

    def line(self, raw: bytes) -> None:
        text = raw.decode(errors="replace").strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            self.tail.append(text)
            return
        if not isinstance(event, dict):
            return
        self.session_id = self.session_id or event.get("sessionID")
        part = event.get("part") or {}
        if event.get("type") == "text" and part.get("text"):
            self.texts.append(part["text"])
        elif event.get("type") == "error":
            self.tail.append(text[:1000])


def pump(runtime, call, log, output: OpencodeOutput, stop_event: threading.Event | None, deadline: float | None = None) -> bool:
    """Copy the process output to the log as it comes. True if cancelled first."""
    fd = call.proc.stdout.fileno()
    while deadline is None or time.monotonic() < deadline:
        if deadline is None and stop_event is not None and stop_event.is_set():
            return True
        exited = runtime.poll(call) is not None
        ready, _, _ = select.select([fd], [], [], 1)
        if not ready:
            if exited:
                break
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        log.write(chunk)
        output.feed(chunk)
        # A tool may retain stdout after opencode exits. Bound only this
        # final drain, so background output cannot keep a finished call alive.
        if exited and deadline is None:
            deadline = time.monotonic() + 5
    output.close()
    return False


def reset_workspace(stage_dir: Path, inputs: list[Path]) -> Path:
    """Empty the workspace and copy in this stage's inputs, read-only."""
    workspace = stage_dir / "workspace"
    if workspace.exists():
        # The agent may have left directories it cannot write; open them up
        # first. Links are not followed, so nothing outside is touched.
        os.chmod(workspace, 0o700)
        for root, dirs, _ in os.walk(workspace):
            for name in dirs:
                path = os.path.join(root, name)
                if not os.path.islink(path):
                    os.chmod(path, 0o700)
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for path in inputs:
        target = workspace / path.name
        shutil.copyfile(path, target)
        target.chmod(0o444)
    return workspace


def run_attempt(
    runtime,
    model_id: str,
    effort: str | None,
    agent: str,
    message: str,
    stage_dir: Path,
    inputs: list[Path],
    expected_first_words: set[str],
    stop_event: threading.Event | None,
) -> str:
    """One opencode call: fresh workspace, one output log, one session export."""
    numbers = [int(p.stem[7:]) for p in stage_dir.glob("output_*.log") if p.stem[7:].isdigit()]
    number = max(numbers, default=0) + 1
    log_path = stage_dir / f"output_{number:03d}.log"
    session_path = stage_dir / f"session_{number:03d}.json"
    output = OpencodeOutput()
    cancelled = False
    code, problem = 1, "call did not finish"
    with open(log_path, "xb", buffering=0) as log:
        host_log(log, f"attempt {number:03d} mode={runtime.mode} agent={agent} model={model_id} variant={effort or '-'}")
        try:
            workspace = reset_workspace(stage_dir, inputs)
            command = ["opencode", "run", "--format", "json", "--model", model_id, "--agent", agent]
            if effort:
                command += ["--variant", effort]
            command.append(message)
            for path in inputs:
                command += ["--file", path.name]  # opencode runs in the workspace
            check_cancelled(stop_event)
            call = runtime.start(command, workspace, model_id, log)
            try:
                cancelled = pump(runtime, call, log, output, stop_event)
            finally:
                cancelled = cancelled or (stop_event is not None and stop_event.is_set())
                # Each step gets its own error handling: an inspection or
                # cleanup error must not skip the available trajectory.
                try:
                    if call.proc.poll() is None or cancelled:
                        runtime.stop(call)
                        pump(runtime, call, log, output, None, deadline=time.monotonic() + 5)
                    code, problem = call.proc.wait(), ""
                except Exception as exc:
                    problem = f"finish failed: {exc}"
                    host_log(log, problem)
                host_log(log, f"{'cancelled, ' if cancelled else ''}exit code {code} {problem}".rstrip())
                save_session(runtime, call, output.session_id, session_path, log)
                try:
                    runtime.cleanup(call)
                except Exception as exc:
                    host_log(log, f"cleanup failed: {exc}; resources may need recovery")
        except Exception as exc:
            host_log(log, f"{type(exc).__name__}: {exc}")
            raise
        if cancelled:
            raise PipelineCancelled("pipeline cancelled")
        text = select_expected_output(output.texts, expected_first_words)
        if code != 0 or problem:
            details = " | ".join(output.tail)[-2000:]
            rejected = f"opencode exited with {code}{', ' + problem if problem else ''} ({log_path}): {details}"
        elif not text:
            rejected = f"empty opencode output ({log_path})"
        elif parse_first_word(text, expected_first_words, "") == "":
            rejected = f"unexpected first word in opencode output: {text.splitlines()[0][:80]}"
        else:
            return text
        # Recorded in this attempt's log, so a retry's reason is not lost.
        host_log(log, f"rejected: {rejected}")
        raise RuntimeError(rejected)


def run_agent(
    runtime,
    model: str,
    effort: str | None,
    agent: str,
    message: str,
    stage_dir: Path,
    inputs: list[Path],
    retries: int,
    expected_first_words: set[str],
    stop_event: threading.Event | None = None,
) -> str:
    """Run one agent turn and return its validated final text."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 2):
        check_cancelled(stop_event)
        try:
            return run_attempt(
                runtime, model, effort, agent, message,
                stage_dir, inputs, expected_first_words, stop_event,
            )  # fmt: skip
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
