"""End-to-end integration tests with real API calls.

These tests hit live PubMed, Semantic Scholar, and Anthropic APIs.
They are marked with @pytest.mark.integration and skipped by default.

Run them explicitly:
    pytest tests/test_integration.py -v -m integration

Requires:
    - ANTHROPIC_API_KEY set in secrets.txt
    - Network access to PubMed and Semantic Scholar
"""

from __future__ import annotations

import logging

import pytest
from lit_review_agent.config import load_secrets
from lit_review_agent.graph import run_review
from lit_review_agent.state import Paper
from lit_review_agent.synthesis import synthesize_paper
from lit_review_agent.tools import dedupe_papers, search_pubmed, search_semantic_scholar

logger = logging.getLogger(__name__)

# Skip all tests in this module if no API key configured
_secrets = load_secrets()
_has_anthropic_key = bool(_secrets.get("ANTHROPIC_API_KEY"))

pytestmark = pytest.mark.integration


def _skip_if_no_key():
    if not _has_anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not configured in secrets.txt")


# ---------------------------------------------------------------------------
# Individual component integration tests
# ---------------------------------------------------------------------------


class TestPubMedLive:
    """Test live PubMed search."""

    def test_search_returns_papers(self):
        papers = search_pubmed(
            "wearable atrial fibrillation deep learning", max_results=3
        )
        # PubMed should return results for this well-known topic
        assert len(papers) > 0
        assert all(isinstance(p, Paper) for p in papers)
        assert all(p.source == "pubmed" for p in papers)
        assert all(p.abstract for p in papers)
        assert all(p.year > 0 for p in papers)
        logger.info("PubMed returned %d papers", len(papers))

    def test_search_empty_query_returns_empty(self):
        papers = search_pubmed("xyznonexistentqueryzyx12345", max_results=5)
        assert papers == []


class TestSemanticScholarLive:
    """Test live Semantic Scholar search."""

    def test_search_returns_papers(self):
        papers = search_semantic_scholar(
            "wearable atrial fibrillation detection", max_results=5
        )
        # S2 may rate-limit — if so, empty is acceptable
        if papers:
            assert all(isinstance(p, Paper) for p in papers)
            assert all(p.source == "semantic_scholar" for p in papers)
            assert all(p.abstract for p in papers)
            logger.info("S2 returned %d papers", len(papers))
        else:
            logger.warning("S2 returned 0 papers (likely rate-limited)")


class TestDeduplicationLive:
    """Test dedup on real overlapping searches."""

    def test_dedup_removes_duplicates_from_overlapping_queries(self):
        q1_papers = search_pubmed("smartwatch ECG atrial fibrillation", max_results=10)
        q2_papers = search_pubmed("wearable ECG AFib detection", max_results=10)

        combined = q1_papers + q2_papers
        deduped = dedupe_papers(combined)

        # Should have fewer than combined (some overlap expected)
        logger.info("Combined: %d, Deduped: %d", len(combined), len(deduped))
        assert len(deduped) <= len(combined)
        # But should still have papers
        assert len(deduped) > 0


class TestSynthesisLive:
    """Test live Claude synthesis on a real paper."""

    def test_synthesize_single_paper(self):
        _skip_if_no_key()

        # Fetch a real paper
        papers = search_pubmed(
            "Apple Watch atrial fibrillation deep learning 2024", max_results=3
        )
        assert len(papers) > 0, "Need at least 1 paper from PubMed"

        paper = papers[0]
        assert paper.task is None  # Not yet synthesized

        result = synthesize_paper(paper)

        # Synthesis should have filled at least some fields
        assert result.title == paper.title  # Original preserved
        assert result.abstract == paper.abstract

        # At least one synthesis field should be filled
        has_extraction = (
            result.task is not None
            or len(result.devices) > 0
            or len(result.sensor_modalities) > 0
            or result.cohort_size is not None
            or len(result.evaluation_metrics) > 0
        )
        assert has_extraction, f"No fields extracted for: {paper.title}"
        logger.info(
            "Synthesized: task=%s, devices=%s, metrics=%s",
            result.task,
            result.devices,
            result.evaluation_metrics,
        )


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------


class TestFullPipelineLive:
    """End-to-end test of run_review() with real APIs.

    This test is expensive (multiple Claude calls) — use sparingly.
    Limited to 1 iteration and narrow query to minimize cost.
    """

    def test_run_review_produces_report(self):
        _skip_if_no_key()

        result = run_review(
            topic="Wearable PPG-based atrial fibrillation detection using deep learning",
            initial_queries=[
                "smartwatch PPG atrial fibrillation deep learning",
            ],
            max_iterations=1,  # Single iteration to minimize cost
        )

        # Should have found papers
        assert len(result["papers"]) > 0, "Pipeline found no papers"
        logger.info("Pipeline found %d papers", len(result["papers"]))

        # Papers should be synthesized (at least some have task filled)
        synthesized = [p for p in result["papers"] if p.task is not None]
        assert len(synthesized) > 0, "No papers were synthesized"
        logger.info("%d/%d papers synthesized", len(synthesized), len(result["papers"]))

        # Should have critic feedback
        assert len(result["critic_feedback"]) > 0

        # Should have a final report
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 200  # Non-trivial report
        logger.info(
            "Report generated: %d chars",
            len(result["final_report"]),
        )
