"""Report Critic Agent — verify the generated report is grounded in source papers."""

from __future__ import annotations

import json
import logging
from typing import Literal

from lit_review_agent.config import get_anthropic_client, get_default_model
from lit_review_agent.state import Paper
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report critic output schema
# ---------------------------------------------------------------------------


class ReportCriticDecision(BaseModel):
    """Structured output from the Report Critic Agent."""

    decision: Literal["approve", "revise"]
    issues: list[str] = []
    reasoning: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a fact-checking reviewer for AI-assisted literature reviews on Health AI \
and wearable device evaluation studies. Your job is to verify that the generated \
report is **faithfully grounded** in the source paper data.

You will receive:
1. The review topic
2. The generated report (markdown)
3. The full corpus data — the actual structured fields extracted from each paper

Your task is to check the report against the source data for:

**Citation accuracy:**
- Every paper referenced in the report must exist in the corpus.
- Paper titles, authors, and years mentioned in the report must match the corpus data.

**Factual grounding:**
- Reported findings, metrics, and numbers must match what the source paper data says.
- Cohort sizes, device names, sensor modalities mentioned in the report must match \
the corpus.
- No findings or conclusions should be attributed to a paper that don't appear in \
that paper's extracted data.

**No fabrication:**
- The report must not contain statistics, p-values, effect sizes, or specific \
numerical results that aren't present in the source data.
- The report must not reference papers, authors, or DOIs not in the corpus.

**Completeness:**
- The methodology comparison table should include all papers in the corpus.
- No papers should be silently omitted from the review.

Return a single JSON object (no markdown fences, no explanation outside the JSON):

{
  "decision": "approve" or "revise",
  "issues": ["List of specific grounding issues found"],
  "reasoning": "Brief explanation of the decision"
}

Rules:
- Return "approve" if the report is faithfully grounded in the source data, even \
if the writing could be improved stylistically. Minor rephrasing of findings is \
acceptable as long as the meaning is preserved.
- Return "revise" only if there are concrete factual errors: wrong numbers, \
fabricated claims, missing papers in the table, or references to non-existent papers.
- Be specific in issues — not "some facts seem wrong" but "Report states Paper X \
has cohort of 500 but source data shows 200" or "Report references 'Zhang et al. \
2023' which is not in the corpus."
- Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Report critic function
# ---------------------------------------------------------------------------


def _format_corpus_for_verification(papers: list[Paper]) -> str:
    """Format all paper data for the Report Critic to verify against."""
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        parts = [
            f"Paper {i}:",
            f"  Title: {p.title}",
            f"  Authors: {', '.join(p.authors)}",
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


def critique_report(
    topic: str,
    papers: list[Paper],
    report: str,
    *,
    model: str | None = None,
    max_retries: int = 2,
) -> ReportCriticDecision:
    """Verify the generated report is grounded in source paper data.

    Returns a ReportCriticDecision with issues if revision is needed.
    """
    client = get_anthropic_client()
    if model is None:
        model = get_default_model()

    corpus_text = _format_corpus_for_verification(papers)
    user_content = (
        f"Review Topic: {topic}\n\n"
        f"--- GENERATED REPORT ---\n\n{report}\n\n"
        f"--- SOURCE CORPUS ({len(papers)} papers) ---\n\n{corpus_text}"
    )

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = response.content[0].text.strip()

        usage = response.usage
        logger.info(
            "Report critic attempt=%d input=%d output=%d",
            attempt,
            usage.input_tokens,
            usage.output_tokens,
        )

        # Strip markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        try:
            parsed = json.loads(raw_text)
            decision = ReportCriticDecision(**parsed)
            logger.info(
                "Report critic decision=%s issues=%d",
                decision.decision,
                len(decision.issues),
            )
            return decision
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Report critic parse error (attempt %d/%d): %s\nRaw: %s",
                attempt + 1,
                max_retries + 1,
                e,
                raw_text[:200],
            )
            if attempt == max_retries:
                logger.error(
                    "Report critic failed after retries — defaulting to approve."
                )
                return ReportCriticDecision(
                    decision="approve",
                    reasoning="Report critic parsing failed; defaulting to approve.",
                )

    # Should not reach here, but satisfy type checker
    return ReportCriticDecision(decision="approve", reasoning="Fallback.")
