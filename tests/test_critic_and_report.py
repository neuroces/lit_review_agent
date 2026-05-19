"""Unit tests for Critic and Report agents.

All Anthropic API calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from lit_review_agent.critic import (
    _format_corpus_summary,
    CriticDecision,
    critique_corpus,
)
from lit_review_agent.report import _format_papers_for_report, generate_report
from lit_review_agent.state import Paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    title: str = "Test Paper",
    task: str | None = "AFib detection",
    devices: list[str] | None = None,
    sensor_modalities: list[str] | None = None,
    cohort_size: int | None = 100,
    split_strategy: str | None = "participant-level",
    evaluation_metrics: list[str] | None = None,
    year: int = 2024,
    doi: str | None = "10.1234/test",
) -> Paper:
    return Paper(
        source="pubmed",
        doi=doi,
        pmid="12345",
        title=title,
        authors=["Smith J"],
        year=year,
        abstract="Test abstract.",
        task=task,
        devices=devices or ["Apple Watch"],
        sensor_modalities=sensor_modalities or ["PPG"],
        cohort_size=cohort_size,
        split_strategy=split_strategy,
        evaluation_metrics=evaluation_metrics or ["sensitivity", "specificity"],
    )


def _mock_anthropic_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    mock.usage = MagicMock(
        input_tokens=500,
        output_tokens=200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return mock


# ---------------------------------------------------------------------------
# CriticDecision schema
# ---------------------------------------------------------------------------


class TestCriticDecision:
    def test_approve(self):
        d = CriticDecision(decision="approve", reasoning="Good coverage.")
        assert d.decision == "approve"
        assert d.gaps == []
        assert d.suggested_queries == []

    def test_refine_with_gaps(self):
        d = CriticDecision(
            decision="refine",
            gaps=["No sleep staging papers", "All small cohorts"],
            suggested_queries=["sleep staging wearable validation"],
            reasoning="Missing key task families.",
        )
        assert d.decision == "refine"
        assert len(d.gaps) == 2

    def test_invalid_decision_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            CriticDecision(decision="maybe")


# ---------------------------------------------------------------------------
# Critic: _format_corpus_summary
# ---------------------------------------------------------------------------


class TestFormatCorpusSummary:
    def test_formats_papers(self):
        papers = [_make_paper(title="Paper A"), _make_paper(title="Paper B")]
        summary = _format_corpus_summary(papers)
        assert "Paper A" in summary
        assert "Paper B" in summary
        assert "AFib detection" in summary
        assert "Apple Watch" in summary

    def test_handles_empty_fields(self):
        paper = _make_paper(
            task=None,
            devices=[],
            sensor_modalities=[],
            cohort_size=None,
            split_strategy=None,
            evaluation_metrics=[],
        )
        summary = _format_corpus_summary([paper])
        assert "not extracted" in summary
        assert "not specified" in summary
        assert "unknown" in summary


# ---------------------------------------------------------------------------
# Critic: critique_corpus (mocked)
# ---------------------------------------------------------------------------


class TestCritiqueCorpus:
    @patch("lit_review_agent.critic.get_anthropic_client")
    def test_returns_approve(self, mock_get_client):
        response_json = json.dumps(
            {
                "decision": "approve",
                "gaps": [],
                "suggested_queries": [],
                "reasoning": "Corpus covers major task families.",
            }
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            response_json
        )
        mock_get_client.return_value = mock_client

        papers = [_make_paper(title=f"Paper {i}") for i in range(5)]
        result = critique_corpus("Health AI wearable benchmarking", papers)

        assert result.decision == "approve"
        assert result.gaps == []

    @patch("lit_review_agent.critic.get_anthropic_client")
    def test_returns_refine_with_queries(self, mock_get_client):
        response_json = json.dumps(
            {
                "decision": "refine",
                "gaps": ["No sleep staging studies"],
                "suggested_queries": ["sleep staging wearable deep learning"],
                "reasoning": "Missing sleep domain.",
            }
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            response_json
        )
        mock_get_client.return_value = mock_client

        result = critique_corpus("test topic", [_make_paper()])
        assert result.decision == "refine"
        assert "sleep staging" in result.suggested_queries[0]

    @patch("lit_review_agent.critic.get_anthropic_client")
    def test_defaults_to_approve_on_parse_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("not json")
        mock_get_client.return_value = mock_client

        result = critique_corpus("test", [_make_paper()], max_retries=1)
        assert result.decision == "approve"
        assert (
            "failed" in result.reasoning.lower()
            or "fallback" in result.reasoning.lower()
        )

    @patch("lit_review_agent.critic.get_anthropic_client")
    def test_handles_markdown_fences(self, mock_get_client):
        fenced = '```json\n{"decision": "approve", "gaps": [], "reasoning": "OK"}\n```'
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(fenced)
        mock_get_client.return_value = mock_client

        result = critique_corpus("test", [_make_paper()])
        assert result.decision == "approve"


# ---------------------------------------------------------------------------
# Report: _format_papers_for_report
# ---------------------------------------------------------------------------


class TestFormatPapersForReport:
    def test_includes_all_fields(self):
        paper = _make_paper(title="AFib Study", doi="10.1/afib")
        text = _format_papers_for_report([paper])
        assert "AFib Study" in text
        assert "10.1/afib" in text
        assert "Apple Watch" in text
        assert "PPG" in text
        assert "participant-level" in text

    def test_handles_missing_fields(self):
        paper = _make_paper(
            doi=None,
            task=None,
            devices=[],
            cohort_size=None,
            split_strategy=None,
        )
        text = _format_papers_for_report([paper])
        assert "N/A" in text
        assert "not extracted" in text
        assert "not specified" in text


# ---------------------------------------------------------------------------
# Report: generate_report (mocked)
# ---------------------------------------------------------------------------


class TestGenerateReport:
    @patch("lit_review_agent.report.get_anthropic_client")
    def test_returns_markdown(self, mock_get_client):
        markdown = "# Literature Review\n\n## Introduction\n\nThis is a test report."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(markdown)
        mock_get_client.return_value = mock_client

        papers = [_make_paper(title=f"Paper {i}") for i in range(3)]
        result = generate_report("Health AI wearable benchmarking", papers)

        assert "# Literature Review" in result
        assert isinstance(result, str)

    @patch("lit_review_agent.report.get_anthropic_client")
    def test_passes_all_papers_in_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response("# Report")
        mock_get_client.return_value = mock_client

        papers = [_make_paper(title=f"Paper {i}") for i in range(5)]
        generate_report("test topic", papers)

        # Verify all papers appear in the user message
        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        for i in range(5):
            assert f"Paper {i}" in user_msg
