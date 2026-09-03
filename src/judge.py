"""Recommend stage 5 -- Judge (paper section 3.3, "Judging").

Checks each claimed resolution for correctness: whether it addresses the
statement it targets, whether the argument is complete, and whether every step
holds. Emits PASS, FAIL, or KNOWN (correct but already in the literature).

Runs `--judges` independent passes over the same solution and aggregates them
conservatively: any FAIL fails, any KNOWN with no FAIL demotes to known, and a
PASS needs every judge to agree.

Because one FAIL settles the verdict, the loop stops at the first one. The
verdict is identical either way, so this only saves the judges whose answers
could not have mattered.

Can run straight after Solve, or as a separate pass over an earlier Solve
output -- it rebuilds the workspace from the persisted record, so re-judging
does not re-run the prover.
"""

import threading
import time
from pathlib import Path
from typing import Any

from src.agent import (
    JUDGE_AGENT,
    JUDGE_WORDS,
    aggregate_judge_verdict,
    classify_result,
    parse_first_word,
    restore_workspace,
    run_agent,
)
from src.prompts import JUDGE_USER_PROMPT



def judge_one(
    item: dict[str, Any],
    model: str,
    effort: str | None,
    body: str,
    work_root: Path,
    retries: int,
    judges: int = 1,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    work_dir = restore_workspace(item, body, work_root)
    input_path = work_dir / "input.json"
    solution_path = work_dir / "solution.md"

    judge_results = []
    for judge_id in range(1, max(judges, 1) + 1):
        judge_started_at = time.time()
        try:
            judge_text = run_agent(
                model,
                effort,
                JUDGE_AGENT,
                JUDGE_USER_PROMPT,
                work_dir,
                retries,
                JUDGE_WORDS,
                [input_path, solution_path],
                stop_event,
            )
            verdict = parse_first_word(judge_text, JUDGE_WORDS, "FAIL")
            judge_error = False
        except Exception as exc:
            verdict = "FAIL"
            judge_text = f"FAIL\nJudge {judge_id} failed: {exc}"
            judge_error = True
        duration = time.time() - judge_started_at
        trace_path = work_dir / f"judge_{judge_id:02d}.md"
        trace_path.write_text(judge_text, encoding="utf-8")
        judge_results.append(
            {
                "judge_id": judge_id,
                "verdict": verdict,
                "duration": round(duration, 3),
                "judgement": judge_text,
                "error": judge_error,
                "trace_path": str(trace_path),
            }
        )
        # A single FAIL decides the verdict, so the remaining judges cannot
        # change it. Stopping here gives the same verdict for less compute.
        if verdict == "FAIL":
            break

    verdict = aggregate_judge_verdict([result["verdict"] for result in judge_results])
    judgement = "\n\n".join(
        f"## Judge {result['judge_id']}: {result['verdict']}\n\n{result['judgement']}"
        for result in judge_results
    )
    # Grade reads judge.md, so write it here whether or not Grade runs inline.
    (work_dir / "judge.md").write_text(judgement or "(no judge output)", encoding="utf-8")

    return {
        **item,
        "verdict": verdict,
        "judgement": judgement,
        "judge_results": judge_results,
        "judge_error": any(result.get("error") for result in judge_results),
        "result": classify_result(item.get("source", "NONE"), verdict),
        "work_dir": str(work_dir),
        "judge_duration": round(time.time() - started_at, 3),
    }



def needs_grading(item: dict[str, Any]) -> bool:
    """Only accepted new results are graded."""
    return item.get("result") == "new"
