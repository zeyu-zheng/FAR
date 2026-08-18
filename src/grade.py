"""Recommend stage 6 -- Grade (paper section 3.3, "Recommending for review").

Takes the outcomes that passed Judge and sorts each as already known, new but
too minor to stand alone, or substantial enough to publish. Deciding the first
requires a fresh literature search: an existing resolution may have escaped
both earlier stages.

    KNOWN  already in the literature
    TYPE1  new but too minor to stand alone
    TYPE2  substantial enough for a standalone paper
    TYPE3  strong enough for a top journal

TYPE2 and TYPE3 form the artifact set A that goes to expert review.

Like Judge, this can run inline or as a separate pass over an earlier run. The
separate pass is the cheap one: it rebuilds the workspace from the persisted
record and never re-runs the prover or the judges.
"""

import threading
import time
from pathlib import Path
from typing import Any

from src.agent import (
    GRADER_AGENT,
    QUALITY_LABELS,
    QUALITY_WORDS,
    parse_first_word,
    restore_workspace,
    run_agent,
)
from src.prompts import GRADER_USER_PROMPT

ARTIFACT_QUALITIES = {"type2", "type3"}



def grade_one(
    item: dict[str, Any],
    model: str,
    effort: str | None,
    body: str,
    work_root: Path,
    retries: int,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    work_dir = restore_workspace(item, body, work_root)
    try:
        grade_text = run_agent(
            model,
            effort,
            GRADER_AGENT,
            GRADER_USER_PROMPT,
            work_dir,
            retries,
            QUALITY_WORDS,
            [work_dir / "input.json", work_dir / "solution.md", work_dir / "judge.md"],
            stop_event,
        )
        quality_word = parse_first_word(grade_text, QUALITY_WORDS, "")
        quality = QUALITY_LABELS.get(quality_word, "ungraded")
    except Exception as exc:  # noqa: BLE001 - record grader failure for manual triage
        grade_text = f"ERROR\nGrader failed: {exc}"
        quality_word = "ERROR"
        quality = "error"

    grade_path = work_dir / "grade.md"
    grade_path.write_text(grade_text, encoding="utf-8")
    return {
        **item,
        "quality": quality,
        "quality_word": quality_word,
        "quality_rationale": grade_text,
        "quality_trace_path": str(grade_path),
        "work_dir": str(work_dir),
        "quality_duration": round(time.time() - started_at, 3),
    }



def is_artifact(item: dict[str, Any]) -> bool:
    """Membership in the artifact set A put forward for expert review."""
    return item.get("quality") in ARTIFACT_QUALITIES
