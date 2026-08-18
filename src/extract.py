"""Find stage 2 -- Extract (paper section 3.1, "Extracting and recovering conjectures").

Extracts explicit unresolved statements from the papers that survive Label. A
paper may yield several or none. Deliberately permissive beyond two exclusions
(future work posing no specific question; statements resolved in the same paper),
since a statement passed over here cannot be recovered later while a spurious
one is dropped by Check.
"""

import asyncio
from typing import Any

from src.prompts import EXTRACT_PROMPT, JSON_ONLY_SYSTEM
from src.schemas import parse_extract_result, validate_extract_result
from src.utils import LLMClient, tail

async def extract_once(body: str, client: LLMClient) -> dict[str, Any]:
    raw = await client.complete(
        [
            {"role": "system", "content": JSON_ONLY_SYSTEM},
            {"role": "user", "content": EXTRACT_PROMPT.format(text=body)},
        ],
        validate_result=validate_extract_result,
        max_validation_retries=2,
    )
    return parse_extract_result(raw)


def process_row(
    row: dict[str, Any],
    body: str,
    client: LLMClient,
) -> dict[str, Any]:
    merged = dict(row)
    merged.update(asyncio.run(extract_once(tail(body), client)))
    return merged


def has_candidates(item: dict[str, Any]) -> bool:
    return bool(item.get("has_open_conjecture")) and bool(item.get("conjectures"))
