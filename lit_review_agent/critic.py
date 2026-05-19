"""Critic Agent — review corpus coverage and decide whether to loop back for more papers."""

from __future__ import annotations

import json
import logging
from typing import Literal

from lit_review_agent.config import get_anthropic_client, get_default_model
from lit_review_agent.state import Paper
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Critic output schema
# ---------------------------------------------------------------------------


class CriticDecision(BaseModel):
    """Structured output from the Critic Agent."""

    decision: Literal["refine", "approve"]
    gaps: list[str] = []
    suggested_queries: list[str] = []
    reasoning: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a systematic review critic specializing in Health AI and wearable device \
evaluation studies. Your job is to assess whether a corpus of papers provides \
adequate coverage for a literature review on the given topic.

You will receive:
1. The review topic
2. A summary of all papers currently in the corpus (title, year, task, devices, \
sensor modalities, cohort size, split strategy, evaluation metrics)

Evaluate coverage along these wearable-specific axes:

**Task families:**
- Are the major Health AI task families represented? (cardiovascular/AFib, sleep \
staging, human activity recognition, metabolic/glucose, mental health/stress, \
gait/falls, vital signs estimation)

**Device diversity:**
- Is there a mix of consumer-grade (smartwatch, fitness tracker) vs. medical-grade \
(Holter, chest patch, clinical wearable) devices?

**Methodological rigor:**
- Are participant-level data splits represented (vs. naive observation-level splits \
that cause data leakage)?
- Is there both retrospective and prospective/external validation work?
- Are diverse evaluation metrics reported (not just accuracy)?

**Demographics and generalizability:**
- Are demographic subgroup analyses reported (skin tone for PPG, age, sex, BMI)?
- Are cohort sizes adequate (not all small pilot studies)?

**Temporal coverage:**
- Is the corpus spanning recent years (2018+) with good representation?

Return a single JSON object (no markdown fences, no explanation outside the JSON):

{
  "decision": "refine" or "approve",
  "gaps": ["List of specific coverage gaps identified"],
  "suggested_queries": ["PubMed/S2 search queries to fill the gaps"],
  "reasoning": "Brief explanation of the decision"
}

Rules:
- Return "approve" if the corpus has reasonable breadth across the axes above, \
even if not perfect. A lit review doesn't need exhaustive coverage.
- Return "refine" only if there are clear, significant gaps that specific queries \
could address. Include concrete suggested_queries.
- Be specific in gaps — not "needs more papers" but "no papers on sleep staging \
with consumer wearables" or "all studies use observation-level splits."
- Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Critic function
# ---------------------------------------------------------------------------


def _format_corpus_summary(papers: list[Paper]) -> str:
    """Format papers into a concise summary for the Critic."""
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        parts = [
            f"{i}. [{p.year}] {p.title}",
            f"   Task: {p.task or 'not extracted'}",
            f"   Devices: {', '.join(p.devices) if p.devices else 'not specified'}",
            f"   Sensors: {', '.join(p.sensor_modalities) if p.sensor_modalities else 'not specified'}",
            f"   Cohort: {p.cohort_size or 'unknown'} participants",
            f"   Split: {p.split_strategy or 'not specified'}",
            f"   Metrics: {', '.join(p.evaluation_metrics) if p.evaluation_metrics else 'not specified'}",
        ]
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def critique_corpus(
    topic: str,
    papers: list[Paper],
    *,
    model: str | None = None,
    max_retries: int = 2,
) -> CriticDecision:
    """Review the corpus and decide whether to refine or approve.

    Returns a CriticDecision with gaps and suggested queries if refinement needed.
    """
    client = get_anthropic_client()
    if model is None:
        model = get_default_model()

    corpus_summary = _format_corpus_summary(papers)
    user_content = (
        f"Review Topic: {topic}\n\n"
        f"Number of papers: {len(papers)}\n\n"
        f"Corpus:\n{corpus_summary}"
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
            "Critic attempt=%d input=%d output=%d",
            attempt,
            usage.input_tokens,
            usage.output_tokens,
        )

        # Strip markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        try:
            parsed = json.loads(raw_text)
            decision = CriticDecision(**parsed)
            logger.info(
                "Critic decision=%s gaps=%d suggested_queries=%d",
                decision.decision,
                len(decision.gaps),
                len(decision.suggested_queries),
            )
            return decision
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Critic parse error (attempt %d/%d): %s\nRaw: %s",
                attempt + 1,
                max_retries + 1,
                e,
                raw_text[:200],
            )
            if attempt == max_retries:
                logger.error("Critic failed after retries — defaulting to approve.")
                return CriticDecision(
                    decision="approve",
                    reasoning="Critic parsing failed; defaulting to approve to avoid infinite loop.",
                )

    # Should not reach here, but satisfy type checker
    return CriticDecision(decision="approve", reasoning="Fallback.")
