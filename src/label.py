"""Find stage 1 -- Label (paper section 3.1, "Finding relevant papers via labeling").

A mathematician fixes a research direction before the run, at whatever
granularity they need. This stage reads each paper and decides whether it lies
in that direction; the papers that do go on to Extract.

The direction is free text, so the same code serves "combinatorics", "extremal
graph theory", or "additive number theory over function fields". It is applied
as written -- the granularity is the user's choice, not the model's.

Runs the cheapest model in the pipeline: it sees the entire corpus.
"""

import asyncio
from typing import Any

from src.prompts import JSON_ONLY_SYSTEM, LABEL_PROMPT
from src.schemas import parse_label_result, validate_label_result
from src.utils import PAPER_FIELD, LLMClient, head, paper_body


async def label_once(row: dict[str, Any], client: LLMClient, direction: str) -> dict[str, Any]:
    body = head(paper_body(row))
    raw = await client.complete(
        [
            {"role": "system", "content": JSON_ONLY_SYSTEM},
            {"role": "user", "content": LABEL_PROMPT.format(direction=direction.strip(), text=body)},
        ],
        validate_result=validate_label_result,
        max_validation_retries=2,
    )
    return parse_label_result(raw)


def process_row(
    row: dict[str, Any],
    client: LLMClient,
    direction: str,
) -> dict[str, Any]:
    result = asyncio.run(label_once(row, client, direction))
    # The corpus row is not carried forward: records reference it by index, and
    # every stage that needs the text reads it back from the corpus.
    merged = {key: value for key, value in row.items() if key != PAPER_FIELD}
    merged.update(result)
    return merged


def keeps(item: dict[str, Any]) -> bool:
    """Papers that go on to Extract."""
    return item.get("in_direction") is True
