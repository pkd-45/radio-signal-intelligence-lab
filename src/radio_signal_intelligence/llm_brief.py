from __future__ import annotations

import csv
from pathlib import Path


def build_candidate_brief(
    ranking_csv: str | Path,
    output_md: str | Path,
    top_n: int = 10,
) -> Path:
    """Create a bounded, auditable candidate brief for optional LLM review.

    No API is called. The generated document contains only structured summary
    fields and explicitly asks an LLM not to alter scores or make discovery claims.
    """
    with Path(ranking_csv).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:top_n]
    if not rows:
        raise ValueError("candidate ranking is empty")

    lines = [
        "# Candidate-review brief",
        "",
        "This file is an optional input to a commercial LLM for concise technical summarisation.",
        "The LLM must not change numerical scores, infer extraterrestrial origin, or replace human review.",
        "",
        "## Review instruction",
        "",
        "Summarise the ranking, identify repeated explanation patterns, and list questions a radio astronomer",
        "should check next. Preserve every number exactly. Treat all candidates as synthetic benchmark items.",
        "",
        "## Structured candidates",
        "",
        "| Rank | Sample | Truth | Predicted | Candidate probability | Triage score | Explanation |",
        "|---:|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {sample_id} | {truth} | {predicted} | {candidate_probability} | "
            "{triage_score} | {explanation} |".format(**row)
        )
    output = Path(output_md)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
