"""Find stage 3 -- Check (paper section 3.1, "Checking validity and status").

For each extracted candidate, searches for later work and labels it `open`,
`solved`, or `invalid`, recording the supporting sources. Open candidates form
the attemptable pool P.

`importance` and `difficulty` are auxiliary: they are recorded for the effort
allocation analysis, not used to filter here.

Runs one model call per candidate, checkpointed individually, so a paper with
many candidates can resume mid-way.
"""

import asyncio
from pathlib import Path
from typing import Any

from src.prompts import CHECK_PROMPT, JSON_ONLY_SYSTEM
from src.schemas import CHECK_RESPONSE_FORMAT, parse_check_result, validate_check_result
from src.utils import LLMClient, append_jsonl, iter_jsonl

async def check_one(candidate: dict[str, Any], client: LLMClient) -> dict[str, Any]:
    raw = await client.complete(
        [
            {"role": "system", "content": JSON_ONLY_SYSTEM},
            {
                "role": "user",
                "content": CHECK_PROMPT.format(
                    title=candidate["title"],
                    authors=candidate["authors"],
                    conjecture_label=candidate["conjecture_label"],
                    conjecture_section=candidate["conjecture_section"],
                    conjecture_text=candidate["conjecture_text"],
                ),
            },
        ],
        validate_result=validate_check_result,
        max_validation_retries=2,
        response_format=CHECK_RESPONSE_FORMAT,
    )
    return parse_check_result(raw)


async def check_row(
    row: dict[str, Any],
    client: LLMClient,
    checkpoint_path: Path | None,
    resume: bool,
) -> dict[str, Any]:
    conjectures = row.get("conjectures") or []
    header = {
        "title": row.get("title", ""),
        "authors": row.get("authors", []),
    }
    if not row.get("has_open_conjecture") or not conjectures:
        return {**header, "has_open_candidate": False, "candidates": []}

    completed = load_checkpoint(checkpoint_path) if (resume and checkpoint_path) else {}
    results = []
    for index, conjecture in enumerate(conjectures, start=1):
        if index in completed:
            results.append(completed[index])
            continue
        candidate = {
            "title": header["title"],
            "authors": header["authors"],
            "conjecture_label": conjecture.get("conjecture_label", ""),
            "conjecture_text": conjecture.get("conjecture_text", ""),
            "conjecture_section": conjecture.get("conjecture_section", ""),
        }
        parsed = await check_one(candidate, client)
        result = {
            "candidate_index": index,
            "conjecture_label": candidate["conjecture_label"],
            "conjecture_text": candidate["conjecture_text"],
            "conjecture_section": candidate["conjecture_section"],
            **parsed,
        }
        results.append(result)
        if checkpoint_path is not None:
            append_checkpoint(checkpoint_path, result)

    if checkpoint_path is not None:
        clear_checkpoint(checkpoint_path)
    results.sort(key=lambda item: (item["status"] != "open", -item["importance"], item["difficulty"]))
    return {
        **header,
        "has_open_candidate": any(item["status"] == "open" for item in results),
        "candidates": results,
    }


def process_row(
    row: dict[str, Any],
    client: LLMClient,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    return asyncio.run(check_row(row, client, checkpoint_path, resume))


# ── Per-candidate checkpoints ───────────────────────────────────────────────

# One file per paper, deleted once the paper is finished. A single shared file
# would have to be searched for every row, and would keep growing after the
# candidates in it stopped mattering.


def checkpoint_path_for(data_dir: Path, row_index: int) -> Path:
    return data_dir / "checkpoints" / "check" / f"{row_index}.jsonl"


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    """The candidates already done for one paper, by candidate index."""
    return {int(item["candidate_index"]): item for item in iter_jsonl(path)}


def append_checkpoint(path: Path, candidate: dict[str, Any]) -> None:
    # Only this paper's worker writes here, but append_jsonl also creates the
    # directory and keeps the write whole.
    append_jsonl(path, candidate)


def clear_checkpoint(path: Path) -> None:
    path.unlink(missing_ok=True)


# ── Pool ────────────────────────────────────────────────────────────────────


def open_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The candidates from one paper that enter the attemptable pool P."""
    tasks = []
    for number, candidate in enumerate(row.get("candidates") or [], start=1):
        if not isinstance(candidate, dict):
            continue
        text = str(candidate.get("conjecture_text") or "").strip()
        if not text:
            continue
        if candidate.get("status") not in {None, "", "open"}:
            continue
        tasks.append(
            {
                "row_index": row.get("row_index"),
                "title": row.get("title") or "Untitled",
                "authors": row.get("authors") or [],
                "candidate_index": candidate.get("candidate_index", number),
                "conjecture_text": text,
                "conjecture_label": candidate.get("conjecture_label"),
                "conjecture_section": candidate.get("conjecture_section"),
                "importance": candidate.get("importance"),
                "difficulty": candidate.get("difficulty"),
                "reason": candidate.get("reason"),
                "sources": candidate.get("sources") or [],
            }
        )
    return tasks
