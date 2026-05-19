"""Synthesis Agent — extract structured fields from paper abstracts using Claude."""

from __future__ import annotations

import json
import logging

from lit_review_agent.config import get_anthropic_client, get_default_model
from lit_review_agent.state import Paper
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction schema (subset of Paper fields filled by synthesis)
# ---------------------------------------------------------------------------


class SynthesisExtraction(BaseModel):
    """Structured extraction from a single paper abstract."""

    task: str | None = None
    devices: list[str] = []
    sensor_modalities: list[str] = []
    cohort_size: int | None = None
    cohort_demographics: str | None = None
    reference_standard: str | None = None
    split_strategy: str | None = None
    evaluation_metrics: list[str] = []
    key_findings: str | None = None
    limitations: str | None = None
    quality_notes: str | None = None


# ---------------------------------------------------------------------------
# System prompt (cacheable — same across all papers)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a biomedical research synthesis assistant specializing in Health AI \
and wearable device evaluation studies.

Given a paper's title, authors, year, and abstract, extract structured information \
into a JSON object. Extract ONLY what is explicitly stated or clearly implied in \
the abstract. Never hallucinate or infer from prior knowledge.

Return a single JSON object (no markdown fences, no explanation) with these fields:

{
  "task": "The primary ML/AI task (e.g., 'AFib detection', 'sleep staging', 'HAR', 'stress detection', 'fall detection', 'heart rate estimation'). null if unclear.",
  "devices": ["List of specific devices or device types mentioned (e.g., 'Apple Watch', 'Fitbit', 'smartwatch', 'chest-worn accelerometer'). Empty list if none."],
  "sensor_modalities": ["Sensor types used (e.g., 'PPG', 'ECG', 'accelerometer', 'gyroscope', 'EDA'). Empty list if none."],
  "cohort_size": "Number of participants/subjects as an integer. null if not stated.",
  "cohort_demographics": "Summary of demographics mentioned (age, sex, race/ethnicity, BMI, comorbidities). null if not stated.",
  "reference_standard": "The ground-truth or reference method (e.g., '12-lead ECG', 'polysomnography', 'self-report', 'expert annotation'). null if not stated.",
  "split_strategy": "Data splitting approach (e.g., 'participant-level', 'observation-level', 'temporal split', 'leave-one-subject-out', 'k-fold', 'external validation'). null if not stated.",
  "evaluation_metrics": ["Metrics reported (e.g., 'accuracy', 'sensitivity', 'specificity', 'AUROC', 'F1', 'MAPE', 'RMSE'). Empty list if none."],
  "key_findings": "One-sentence summary of the main result. null if unclear.",
  "limitations": "Limitations mentioned by the authors. null if none stated.",
  "quality_notes": "Your brief assessment of methodological quality based on what's reported (e.g., 'No mention of split strategy — possible data leakage', 'Small cohort', 'External validation included'). null if nothing notable."
}

Rules:
- Return ONLY the JSON object. No preamble, no markdown code fences.
- Use null for fields where information is not available in the abstract.
- Use empty lists [] for list fields where no items are found.
- For cohort_size, extract the number of participants, not observations/recordings.
- Be precise with device names and sensor modalities.
"""


# ---------------------------------------------------------------------------
# Synthesis function
# ---------------------------------------------------------------------------


def synthesize_paper(
    paper: Paper,
    *,
    model: str | None = None,
    max_retries: int = 2,
) -> Paper:
    """Extract structured fields from a paper's abstract using Claude.

    Returns a new Paper with synthesis fields populated.
    Uses prompt caching for the system prompt (same across all papers).
    Retries on JSON parse or Pydantic validation failures.
    """
    client = get_anthropic_client()
    if model is None:
        model = get_default_model()

    user_content = (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:10])}\n"
        f"Year: {paper.year}\n"
        f"Abstract: {paper.abstract}"
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

        # Log cache stats
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_create = getattr(usage, "cache_creation_input_tokens", 0)
        logger.info(
            "Synthesis [%s] attempt=%d input=%d cache_read=%d cache_create=%d output=%d",
            paper.pmid or paper.doi or paper.title[:30],
            attempt,
            usage.input_tokens,
            cache_read,
            cache_create,
            usage.output_tokens,
        )

        # Strip markdown fences if the model wraps anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        try:
            parsed = json.loads(raw_text)
            extraction = SynthesisExtraction(**parsed)
            break
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Synthesis parse error (attempt %d/%d): %s\nRaw: %s",
                attempt + 1,
                max_retries + 1,
                e,
                raw_text[:200],
            )
            if attempt == max_retries:
                logger.error(
                    "Synthesis failed after %d retries for: %s",
                    max_retries + 1,
                    paper.title[:60],
                )
                return paper  # return unmodified paper

    # Merge extraction into a copy of the paper
    updated = paper.model_copy(update=extraction.model_dump(exclude_none=True))
    return updated


def synthesize_papers(
    papers: list[Paper],
    *,
    model: str | None = None,
) -> list[Paper]:
    """Synthesize all papers sequentially. Returns list with synthesis fields filled."""
    results: list[Paper] = []
    for i, paper in enumerate(papers):
        logger.info(
            "Synthesizing paper %d/%d: %s", i + 1, len(papers), paper.title[:60]
        )
        result = synthesize_paper(paper, model=model)
        results.append(result)
    return results
