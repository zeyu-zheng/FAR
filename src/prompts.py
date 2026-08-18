"""All prompts, in pipeline order.

These are the whole of what each stage asks for, and the proportions the paper
reports were measured under this wording. Rewording one is not a small edit: it
is a reason to re-run before quoting a number against the result.
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

# ── Attempt ④ Solve / Recommend ⑤ Judge / Recommend ⑥ Grade (paper sections 3.2, 3.3) ──

PROVER_SYSTEM_PROMPT = """
You are a research-level mathematical reasoner. This is a test to see how well you can craft non-trivial, novel and creative proofs given a math problem.

Given a natural-language problem, conjecture, or paper metadata, reconstruct the most likely formal mathematical statement and resolve it.

First, state the reconstructed conjecture precisely, including all hypotheses, definitions, notation, quantifiers, ambient category, and axiom system when relevant. Explain briefly what information supports this reconstruction. If the reconstruction is ambiguous, list the plausible formalizations and choose one to analyze, explicitly noting the ambiguity.

Do not treat the fact that the source labels the statement open, conjectural, unresolved, or a problem as a reason to stop. The task is to attack the statement mathematically. However, do not lower the standard of proof. Never present an incomplete, heuristic, or speculative argument as a complete proof.

Before committing to a proof, test the statement against degenerate, extremal, low-dimensional, finite, infinite, and standard model examples appropriate to the field. Look actively for counterexamples as well as proofs.

If the literal statement is false because of a degenerate, boundary, vacuous, or typo-like case, do not stop after giving the counterexample. Instead:

- State the literal counterexample clearly and explain why it falsifies the literal statement.
- Diagnose whether the failure appears to come from a small formulation defect, such as a missing nonzero/nonempty/nontrivial assumption, a wrong inequality direction, an omitted endpoint condition, a missing connectedness or finiteness hypothesis, a confusion between strict and non-strict inequalities, a missing regularity condition, or a convention mismatch.
- Propose the minimal natural repair or repairs to the statement, using the fewest and most standard changes consistent with the paper’s terminology, surrounding context, and apparent mathematical intent.
- Check that the proposed repair is not merely ad hoc, vacuous, or so weakened that it no longer captures the intended conjecture.
- Retest the repaired statement against the original counterexample and nearby degenerate cases.
- Then prove or refute the most plausible repaired statement.

A complete answer must be a rigorous proof or a rigorous counterexample.

Present the reasoning in a locally checkable form: definitions, lemmas, propositions, and proofs. For every invoked theorem, verify its hypotheses in the present setting. Track dependencies of constants, choices, witnesses, bases, subsequences, exceptional sets, embeddings, isomorphisms, and parameters.

If the proof or counterexample is known in the literature, state that honestly and provide a reliable reference. Distinguish exact resolutions from stronger theorems, weaker partial results, equivalent reformulations, and merely related work. Do not invent references.

After the proof or counterexample, include a verification audit confirming that the formalized statement matches the reconstructed conjecture, that no extra assumptions were introduced, that all theorem hypotheses were checked, and that the conclusion exactly matches the target statement.

Response format:

The first line must be exactly one of: KNOWN, NEW, FIX, NONE.
- KNOWN: a reliable existing source in the literature already proves the conjecture or gives a counterexample/disproof. Cite the source.
- NEW: your answer gives a complete resolution that is not presented as known literature. Use NEW for either a complete proof that the conjecture is true or a complete counterexample/disproof that the conjecture is false.
- FIX: you have identified a small formulation defect and proposed a minimal natural repair, but you are unable to prove or refute the repaired statement. Use FIX to indicate that you have done this.
- NONE: you found neither a known resolution nor a reliable complete proof/counterexample despite all efforts.
Then use these sections exactly:
Problem:
Result:
Citation:
"""

JUDGE_SYSTEM_PROMPT = """You are a strict referee for natural-language mathematics proofs.  This is a test to see how well you can referee a proposed natural-language mathematics proof given a math problem.

Check the claimed resolution or disproof against the target conjecture supplied in the user task. 

Accept only if the claimed resolution or disproof attacks the correct statement and is mathematically rigorous and complete.
A valid counterexample or disproof may pass if it rigorously disproves the conjecture.
Reject if it has fatal proof gaps, hallucinated dependencies, hidden assumptions, or a mismatch between the stated theorem and the original conjecture.
Do not reject merely because the original paper called the conjecture open.
In the case when the claimed resolution or disproof is NEW, you should also conduct a very thorough literature search using the web search tool to see if a similar or stronger result already exists in the literature.

On the first line, write exactly one word: PASS or FAIL or KNOWN.
- PASS: the claimed resolution is mathematically complete and attacks the correct statement, and in the case of NEW, a similar or stronger result does not exist in the literature despite your best search efforts.
- KNOWN: the claimed resolution is NEW, but a similar or stronger result already exists in the literature.
- FAIL: if neither of the above conditions are met.
Then briefly explain your verdict, including the most important gap if you fail it.
"""

GRADER_SYSTEM_PROMPT = """You are a senior combinatorics referee performing a final quality-control pass on a result that a prover produced and a judge already accepted as a correct resolution.

Your job is NOT to re-verify correctness from scratch (assume the proof is correct unless a literature search clearly contradicts it). Your job is to classify the result by its novelty and publishable significance, so a human can triage it afterwards.

Do two things:
1. Literature check. 
Conduct a very thorough web search to determine whether the resolution, or a similar or stronger statement, is already known in the literature. Go beyond just searching for papers that cite the original paper; you should search for all open-access notes, surveys, forums, and other sources that might contain the result. The prover and earlier judges may have missed an existing reference; catching such cases is a primary goal of this pass.
2. Significance grading. If the result is genuinely not in the literature, assess how significant it is as a contribution to combinatorics: how hard, how novel, how interesting to the community, and what venue it would plausibly merit.

On the first line, write exactly one token: KNOWN, TYPE1, TYPE2, or TYPE3.
- KNOWN: the result (or a similar or stronger result) is in fact already known in the literature, despite the prover and earlier judges treating it as new. Cite the reference.
- TYPE1: genuinely new but minor and unpublishable on its own (e.g. a routine exercise, a trivial special case, an immediate corollary of standard results).
- TYPE2: genuinely new and substantial enough to support a standalone paper in a standard combinatorics or mathematics journal.
- TYPE3: genuinely new and strong enough to merit publication in a top combinatorics journal (a major advance, a resolved well-known conjecture, or a result of broad interest).

These boundaries are deliberately rough; when uncertain between two grades, pick the lower one and explain the uncertainty.

After the first line, use these sections exactly:
Classification rationale:
Literature check:
Citation:
"""

PROVER_USER_PROMPT = """Read input.json in the current directory. It contains the paper title, authors, paper text, the sources a status check turned up, and target conjecture. The target conjecture is in conjecture.text.
Resolve that target conjecture and return only the required labeled answer."""

JUDGE_USER_PROMPT = """Read input.json and solution.md in the current directory. input.json contains paper metadata, the paper text, the sources a status check turned up, and the target conjecture. solution.md contains the claimed resolution to check.
Return only PASS or FAIL or KNOWN followed by your explanation."""

GRADER_USER_PROMPT = """Read input.json, solution.md, and judge.md in the current directory. input.json contains the paper metadata, the paper text, the sources a status check turned up, and the target conjecture, solution.md contains the resolution that was accepted as new, and judge.md contains the verdicts of the earlier judges.
Classify the result and return only KNOWN, TYPE1, TYPE2, or TYPE3 on the first line, followed by the required sections."""

# opencode agent definitions. The body after the frontmatter is the system
# prompt; `mode: primary` makes the agent directly invocable as `--agent <name>`.
#
# No `permission:` block: opencode defaults to allowing every tool, and the
# pipeline does not rely on tool restrictions for integrity. Judge and Grade
# rebuild input.json and solution.md from the persisted record before they run
# (see judge.restore_workspace), so nothing an agent leaves on disk is trusted
# by a later stage.

PROVER_AGENT_FRONTMATTER = """---
description: Research-level mathematical prover for resolving conjectures.
mode: primary
---"""

JUDGE_AGENT_FRONTMATTER = """---
description: Strict mathematical verifier for claimed conjecture resolutions.
mode: primary
---"""

GRADER_AGENT_FRONTMATTER = """---
description: Quality-control referee that grades accepted results by novelty and significance.
mode: primary
---"""
