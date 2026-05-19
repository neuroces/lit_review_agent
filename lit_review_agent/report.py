"""Report Agent — generate a structured markdown literature review from the synthesized corpus."""

from __future__ import annotations

import logging

from lit_review_agent.config import get_anthropic_client
from lit_review_agent.state import Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a scientific writing assistant specializing in Health AI literature reviews, \
with deep expertise in wearable device evaluation studies.

Given a review topic and a corpus of papers with extracted metadata, produce a \
structured **markdown** literature review. The review should be rigorous, concise, \
and suitable for a technical audience in Health AI / digital health.

## Required sections

1. **Introduction** — State the topic, its relevance, and the scope of this review.

2. **Methods** — Describe the review process: automated literature search across \
PubMed and Semantic Scholar, multi-agent extraction pipeline, coverage analysis via \
a critic loop. Be transparent that this is an AI-assisted review. State the number \
of papers, year range, and sources.

3. **Corpus Summary** — N papers, year range, source breakdown, task distribution.

4. **Methodology Comparison Table** — A markdown table with columns:
   | Paper | Year | Task | Device(s) | Sensor | Cohort N | Reference Standard | Split Strategy | Metrics | Key Finding |

5. **Themes** — Cross-cutting themes synthesized across papers (e.g., device \
heterogeneity, evaluation metric choices, validation approaches).

6. **Gaps and Open Questions** — What is underrepresented or missing in the \
current literature.

7. **Conclusions** — Brief summary of the state of the field and recommendations.

8. **References** — Full list with DOI links where available.

## Rules
- Write in academic style but keep it concise.
- The comparison table should include ALL papers in the corpus.
- Use information ONLY from the provided paper data. Do not hallucinate references \
or findings.
- For the references section, format as: Authors (Year). Title. DOI: link (if available).
- Return ONLY the markdown content, no preamble.
"""


# ---------------------------------------------------------------------------
# Report function
# ---------------------------------------------------------------------------


def _format_papers_for_report(papers: list[Paper]) -> str:
    """Format all paper data as structured text for the Report Agent."""
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        parts = [
            f"Paper {i}:",
            f"  Title: {p.title}",
            f"  Authors: {', '.join(p.authors[:5])}{'...' if len(p.authors) > 5 else ''}",
            f"  Year: {p.year}",
            f"  DOI: {p.doi or 'N/A'}",
            f"  PMID: {p.pmid or 'N/A'}",
            f"  Source: {p.source}",
            f"  Task: {p.task or 'not extracted'}",
            f"  Devices: {', '.join(p.devices) if p.devices else 'not specified'}",
            f"  Sensor modalities: {', '.join(p.sensor_modalities) if p.sensor_modalities else 'not specified'}",
            f"  Cohort size: {p.cohort_size or 'unknown'}",
            f"  Demographics: {p.cohort_demographics or 'not reported'}",
            f"  Reference standard: {p.reference_standard or 'not specified'}",
            f"  Split strategy: {p.split_strategy or 'not specified'}",
            f"  Evaluation metrics: {', '.join(p.evaluation_metrics) if p.evaluation_metrics else 'not specified'}",
            f"  Key findings: {p.key_findings or 'not extracted'}",
            f"  Limitations: {p.limitations or 'not reported'}",
            f"  Quality notes: {p.quality_notes or 'none'}",
        ]
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def generate_report(
    topic: str,
    papers: list[Paper],
    *,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate a structured markdown literature review from the synthesized corpus.

    Returns the full markdown report as a string.
    """
    client = get_anthropic_client()

    corpus_text = _format_papers_for_report(papers)
    user_content = (
        f"Review Topic: {topic}\n\n"
        f"Number of papers: {len(papers)}\n\n"
        f"Full corpus data:\n\n{corpus_text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    report = response.content[0].text.strip()

    usage = response.usage
    logger.info(
        "Report generated: input=%d output=%d tokens",
        usage.input_tokens,
        usage.output_tokens,
    )

    return report
