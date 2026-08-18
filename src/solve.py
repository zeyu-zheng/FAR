"""Attempt stage 4 -- Solve (paper section 3.2, "Attempting for Candidate Resolutions").

Every conjecture in the pool P receives one attempt. The agent searches the
literature first, then attempts a proof or counterexample, and labels the
outcome:

    KNOWN  a credible source already resolves it
    NEW    it produced a complete proof or counterexample of its own
    FIX    the statement as written is defective and the minimal repair it
           proposes cannot be settled
    NONE   none of the above

Only NEW outcomes go on to Judge.

Runs the most capable model in the pipeline: Find has narrowed the pool, so
this stage can afford more compute per item.
"""

import threading
import time
from pathlib import Path
from typing import Any

from src.agent import (
    PROVER_AGENT,
    SOURCE_WORDS,
    parse_first_word,
    prepare_work_dir,
    run_agent,
)
from src.prompts import PROVER_USER_PROMPT


def solve_one(
    task: dict[str, Any],
    model: str,
    effort: str | None,
    body: str,
    work_root: Path,
    retries: int,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one attempt and return the record Judge consumes."""
    started_at = time.time()
    work_dir = prepare_work_dir(task, body, work_root)
    input_path = work_dir / "input.json"

    solution = run_agent(
        model,
        effort,
        PROVER_AGENT,
        PROVER_USER_PROMPT,
        work_dir,
        retries,
        SOURCE_WORDS,
        [input_path],
        stop_event,
    )
    source = parse_first_word(solution, SOURCE_WORDS, "NONE")

    # Persist the solution next to input.json so Judge finds the workspace
    # already populated when it runs straight after this stage.
    if source in {"KNOWN", "NEW"}:
        (work_dir / "solution.md").write_text(solution, encoding="utf-8")

    return {
        # Carried forward so Judge and Grade can rebuild an identical workspace
        # without re-running this stage. See agent.TASK_KEYS.
        **task,
        "source": source,
        "solution": solution,
        "work_dir": str(work_dir),
        "solve_duration": round(time.time() - started_at, 3),
    }


def needs_judging(item: dict[str, Any]) -> bool:
    """Only KNOWN and NEW carry a claimed resolution worth checking."""
    return item.get("source") in {"KNOWN", "NEW"}
