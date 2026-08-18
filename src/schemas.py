"""Output schemas and validators for the three Find stages.

Field names match the JSON blocks in the paper's appendix A. A validator
returns a message rather than raising, and that message goes back to the model
as the reason its answer was rejected, so it is written to be read by one.

Solve, Judge, and Grade return a first-line token rather than JSON; their
parsing lives in agent.py.
"""

from typing import Any

from src.utils import extract_first_json, validate_score

# ── Find ① Label ────────────────────────────────────────────────────────────

LABEL_KEYS = {"in_direction", "comment"}


def parse_label_result(text: str) -> dict[str, Any]:
    obj = extract_first_json(text, LABEL_KEYS)
    if obj is None:
        raise ValueError("no label JSON object found")
    in_direction = obj.get("in_direction")
    comment = obj.get("comment")
    if not isinstance(in_direction, bool):
        raise ValueError("in_direction must be a boolean")
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("comment is missing or empty")
    return {
        "comment": comment.strip(),
        "in_direction": in_direction,
    }


def validate_label_result(text: str) -> dict[str, Any]:
    try:
        parse_label_result(text)
    except Exception as exc:
        return {
            "success": False,
            "message": (
                "Your previous final JSON result was rejected. "
                f"Reason: {exc}. "
                "Reply again with exactly one JSON object containing "
                "comment (a non-empty English string) and in_direction (a JSON boolean)."
            ),
        }
    return {"success": True}


# ── Find ② Extract ──────────────────────────────────────────────────────────

EXTRACT_TOP_LEVEL_KEYS = {"title", "authors", "decision_basis", "has_open_conjecture", "conjectures"}
CONJECTURE_KEYS = {"conjecture_label", "conjecture_text", "conjecture_section"}


def parse_extract_result(text: str) -> dict[str, Any]:
    obj = extract_first_json(text, EXTRACT_TOP_LEVEL_KEYS)
    if obj is None:
        raise ValueError("no conjecture JSON object found")
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    decision_basis = obj.get("decision_basis")
    if not isinstance(decision_basis, str) or not decision_basis.strip():
        raise ValueError("decision_basis must be a non-empty string")

    has_open_conjecture = obj.get("has_open_conjecture")
    if not isinstance(has_open_conjecture, bool):
        raise ValueError("has_open_conjecture must be a boolean")
    return {
        "title": title.strip(),
        "authors": validate_authors(obj.get("authors")),
        "decision_basis": decision_basis.strip(),
        "has_open_conjecture": has_open_conjecture,
        "conjectures": validate_conjectures(obj.get("conjectures"), has_open_conjecture),
    }


def validate_extract_result(text: str) -> dict[str, Any]:
    try:
        parse_extract_result(text)
    except Exception as exc:
        return {
            "success": False,
            "message": (
                "Your previous final JSON result was rejected. "
                f"Reason: {exc}. "
                "Reply again with exactly one JSON object containing "
                "title, authors, decision_basis, has_open_conjecture, and conjectures. "
                "If has_open_conjecture is false, conjectures must be []. "
                "If has_open_conjecture is true, conjectures must list every explicit unresolved statement you found."
            ),
        }
    return {"success": True}


def validate_authors(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("authors must be a non-empty array")
    authors = [author.strip() for author in value if isinstance(author, str) and author.strip()]
    if len(authors) != len(value):
        raise ValueError("authors must contain only non-empty strings")
    return authors


def validate_conjectures(value: Any, has_open_conjecture: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("conjectures must be an array")
    parsed = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"conjectures[{index}] must be an object")
        if not CONJECTURE_KEYS.issubset(item):
            raise ValueError(f"conjectures[{index}] is missing required keys {sorted(CONJECTURE_KEYS)}")
        label = item.get("conjecture_label")
        text = item.get("conjecture_text")
        section = item.get("conjecture_section")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"conjectures[{index}].conjecture_label must be non-empty")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"conjectures[{index}].conjecture_text must be non-empty")
        if not isinstance(section, str):
            raise ValueError(f"conjectures[{index}].conjecture_section must be a string")
        parsed.append({
            "conjecture_label": label.strip(),
            "conjecture_text": text.strip(),
            "conjecture_section": section.strip(),
        })
    if has_open_conjecture and not parsed:
        raise ValueError("has_open_conjecture is true but conjectures is empty")
    if not has_open_conjecture and parsed:
        raise ValueError("has_open_conjecture is false but conjectures is not empty")
    return parsed

# ── Find ③ Check ────────────────────────────────────────────────────────────

CHECK_REQUIRED_KEYS = {
    "sources",
    "reason",
    "status",
    "importance",
    "difficulty",
}

STATUSES = {"open", "solved", "invalid"}

CHECK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "claim": {"type": "string"},
                        },
                        "required": ["title", "url", "claim"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "solved", "invalid"]},
                "importance": {"type": "number"},
                "difficulty": {"type": "number"},
            },
            "required": ["sources", "reason", "status", "importance", "difficulty"],
            "additionalProperties": False,
        },
    },
}


def parse_check_result(text: str) -> dict[str, Any]:
    obj = extract_first_json(text, CHECK_REQUIRED_KEYS)
    if obj is None:
        raise ValueError("no status JSON object found")

    status = obj.get("status")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    reason = str(obj.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason must be a non-empty string")

    sources = validate_sources(obj.get("sources"))
    if status == "solved" and not sources:
        raise ValueError("status is solved but no source was given; a resolution needs its evidence")

    return {
        "sources": sources,
        "reason": reason,
        "status": status,
        "importance": validate_score(obj.get("importance"), "importance"),
        "difficulty": validate_score(obj.get("difficulty"), "difficulty"),
    }


def validate_check_result(text: str) -> dict[str, Any]:
    try:
        parse_check_result(text)
    except Exception as exc:
        return {
            "success": False,
            "message": (
                "Your previous final JSON result was rejected. "
                f"Reason: {exc}. "
                "Reply again with exactly one JSON object. "
                "Ensure status is one of open, solved, or invalid; sources is an array of "
                "objects, non-empty when the status is solved; reason is a non-empty English "
                "string; and importance and difficulty are numbers in [0, 1]."
            ),
        }
    return {"success": True}


def validate_sources(value: Any) -> list[dict[str, str]]:
    """Every source the model gave, kept whole.

    Nothing is dropped silently: these are the evidence for the status, and the
    agents downstream read them as leads. A malformed entry is rejected so the
    model is asked to fix it rather than having it disappear.
    """
    if not isinstance(value, list):
        raise ValueError("sources must be an array")
    sources = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{index}] must be an object")
        sources.append({
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "claim": str(item.get("claim") or "").strip(),
        })
    return sources
