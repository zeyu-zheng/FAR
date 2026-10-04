"""Find prompts and per-task agent messages, in pipeline order.

Agent system prompts live in agents/*.md. The proportions the paper
reports were measured under this wording. Rewording a prompt is a reason to
re-run before quoting a number against the result.
"""

JSON_ONLY_SYSTEM = "Output only one JSON object matching the requested schema."

# ── Find ① Label (paper section 3.1, "Finding relevant papers via labeling") ──

LABEL_PROMPT = """Return one JSON object with this schema:
{{
  "comment": "...",
  "in_direction": false
}}

Rules:
- `comment` must name the paper's primary subject in a few words.
- `in_direction` must be a JSON boolean.
- Use `true` when the paper's primary content lies in the research direction.
- Use `false` when it does not, when the content is not mathematical, or when the paper appears mislabeled.

Research direction: {direction}

Paper content:
{text}
"""

# ── Find ② Extract (paper section 3.1, "Extracting and recovering conjectures") ──

EXTRACT_PROMPT = """Return one JSON object with this schema:
{{
  "title": "...",
  "authors": ["...", "..."],
  "decision_basis": "...",
  "has_open_conjecture": false,
  "conjectures": [
    {{
      "conjecture_label": "...",
      "conjecture_text": "...",
      "conjecture_section": "..."
    }}
  ]
}}

Rules:
- `title` must be a non-empty string.
- `authors` must be a JSON array of non-empty author-name strings.
- `decision_basis` must be one short English sentence.
- `has_open_conjecture` must be a JSON boolean.
- `conjectures` must be a JSON array. If `has_open_conjecture` is false, it must be `[]`.
- Set `has_open_conjecture` to true iff the paper contains at least one explicit unresolved mathematical statement.
- Count these as hits:
  1. labeled `Conjecture` / `Question` / `Open Problem`
  2. sentences with markers like `open question`, `open problem`, `open issue`, `remains unknown whether`, or `we suspect ... although we have been unable to establish ...`
  3. a direct statement that a specific mathematical property, existence claim, or classification problem `still remains an open issue`
- Do NOT count:
  1. generic future work that does not pose a specific mathematical question
  2. results that have already been proved or resolved within the paper itself
- If a sentence says a specific claim or property is `still an open issue`, count it even if it is not written as a formal question.
- If `has_open_conjecture` is true, extract only the explicit unresolved statements themselves, not nearby speculation.
- `conjecture_label` should use the paper's label when present, otherwise use a short fallback like `Unlabeled open problem 1`.
- `conjecture_text` should copy the paper's unresolved statement as faithfully as possible and preserve notation.
- `conjecture_section` should be the visible section/subsection title, or `""` if unavailable.

Paper content:
{text}
"""

# ── Find ③ Check (paper section 3.1, "Checking validity and status") ──

CHECK_PROMPT = """Return one JSON object with this schema:
{{
  "sources": [
    {{"title": "...", "url": "...", "claim": "..."}}
  ],
  "reason": "...",
  "status": "solved",
  "importance": 0.5,
  "difficulty": 0.5
}}

Rules:
- Verify the candidate's current status using current web information.
- `status` must be one of: `open`, `solved`, `invalid`.
- Use `open` when the candidate is a concrete open problem in the source and no credible solved evidence is found.
- Use `solved` when a credible source appears to solve it.
- Use `invalid` when it is not a concrete open problem in the source.
- `sources` should list only sources directly supporting the status.
- each `claim` must be what that source says about this candidate.
- for `solved`, `sources` must name at least one source that resolves the candidate.
- `reason` must be one concise English sentence.
- `importance` must be a number in [0, 1] for the candidate itself: candidates with no substantive mathematical content should be scored 0; Fields-Medal-level problems should be scored 1; most ordinary research problems should follow a roughly normal distribution centered around 0.5.
- `difficulty` must be a number in [0, 1]: solving it would be an unpublishable exercise should be scored 0; solving it would be publishable in a top journal (Annals, Inventiones, JAMS, Acta) should be scored 1; most problems should follow a roughly normal distribution centered around 0.5.
- For `solved` or `invalid`, set `importance` and `difficulty` to 0.

Paper title: {title}
Paper authors: {authors}
Candidate label: {conjecture_label}
Candidate section: {conjecture_section}
Candidate text:
{conjecture_text}
"""

# Agent system prompts live in agents/*.md.
# Per-task user messages remain here.

PROVER_USER_PROMPT = """Read input.json in the current directory. It contains the paper title, authors, paper text, the sources a status check turned up, and target conjecture. The target conjecture is in conjecture.text.
Resolve that target conjecture and return only the required labeled answer."""

JUDGE_USER_PROMPT = """Read input.json and solution.md in the current directory. input.json contains paper metadata, the paper text, the sources a status check turned up, and the target conjecture. solution.md contains the claimed resolution to check.
Return only PASS or FAIL or KNOWN followed by your explanation."""

GRADER_USER_PROMPT = """Read input.json, solution.md, and judge.md in the current directory. input.json contains the paper metadata, the paper text, the sources a status check turned up, and the target conjecture, solution.md contains the resolution that was accepted as new, and judge.md contains the verdicts of the earlier judges.
Classify the result and return only KNOWN, TYPE1, TYPE2, or TYPE3 on the first line, followed by the required sections."""
