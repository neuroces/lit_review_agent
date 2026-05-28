"""Unit tests for the LangGraph graph wiring.

Tests graph construction, node functions, and conditional routing.
All external calls mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from lit_review_agent.graph import (
    build_graph,
    compile_graph,
    critic_node,
    critic_router,
    report_critic_node,
    report_critic_router,
    search_node,
    synthesize_node,
)
from lit_review_agent.state import Paper, ReviewState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    title: str = "Test", task: str | None = None, doi: str | None = "10.1/t"
) -> Paper:
    return Paper(
        source="pubmed",
        doi=doi,
        pmid="123",
        title=title,
        authors=["A"],
        year=2024,
        abstract="Test abstract.",
        task=task,
    )


def _base_state(**overrides) -> ReviewState:
    defaults: ReviewState = {
        "topic": "Health AI wearable benchmarking",
        "search_queries": ["wearable deep learning"],
        "papers": [],
        "critic_feedback": [],
        "iteration": 0,
        "max_iterations": 3,
        "final_report": None,
        "report_critic_feedback": [],
        "report_iteration": 0,
        "max_report_iterations": 2,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_builds_with_all_nodes(self):
        graph = build_graph()
        assert "search" in graph.nodes
        assert "synthesize" in graph.nodes
        assert "critic" in graph.nodes
        assert "report" in graph.nodes
        assert "report_critic" in graph.nodes

    def test_compiles_successfully(self):
        app = compile_graph()
        assert app is not None
        assert type(app).__name__ == "CompiledStateGraph"


# ---------------------------------------------------------------------------
# search_node
# ---------------------------------------------------------------------------


class TestSearchNode:
    @patch("lit_review_agent.graph.time.sleep")
    @patch("lit_review_agent.graph.search_semantic_scholar", return_value=[])
    @patch("lit_review_agent.graph.search_pubmed")
    @patch("lit_review_agent.graph.dedupe_papers", side_effect=lambda x: x)
    def test_searches_all_queries(self, mock_dedup, mock_pm, mock_s2, _sleep):
        mock_pm.return_value = [_make_paper("PM Paper")]
        state = _base_state(search_queries=["q1", "q2"])

        result = search_node(state)

        assert mock_pm.call_count == 2
        assert mock_s2.call_count == 2
        assert len(result["papers"]) == 2  # 2 queries × 1 PM paper each

    @patch("lit_review_agent.graph.time.sleep")
    @patch("lit_review_agent.graph.search_semantic_scholar", return_value=[])
    @patch("lit_review_agent.graph.search_pubmed", return_value=[])
    @patch("lit_review_agent.graph.dedupe_papers", side_effect=lambda x: x)
    def test_preserves_existing_papers(self, mock_dedup, mock_pm, mock_s2, _sleep):
        existing = [_make_paper("Existing")]
        state = _base_state(papers=existing)

        result = search_node(state)
        assert any(p.title == "Existing" for p in result["papers"])


# ---------------------------------------------------------------------------
# synthesize_node
# ---------------------------------------------------------------------------


class TestSynthesizeNode:
    @patch("lit_review_agent.graph.synthesize_papers")
    def test_only_processes_unsynthesized(self, mock_synth):
        done = _make_paper("Done", task="AFib")
        pending = _make_paper("Pending", task=None)
        mock_synth.return_value = [_make_paper("Pending", task="HAR")]

        state = _base_state(papers=[done, pending])
        result = synthesize_node(state)

        # Should only pass the unsynthesized paper
        mock_synth.assert_called_once()
        passed_papers = mock_synth.call_args[0][0]
        assert len(passed_papers) == 1
        assert passed_papers[0].title == "Pending"

        # Result should contain both
        assert len(result["papers"]) == 2

    @patch("lit_review_agent.graph.synthesize_papers")
    def test_skips_when_all_done(self, mock_synth):
        papers = [_make_paper("Done", task="AFib")]
        state = _base_state(papers=papers)

        result = synthesize_node(state)
        mock_synth.assert_not_called()
        assert len(result["papers"]) == 1


# ---------------------------------------------------------------------------
# critic_node
# ---------------------------------------------------------------------------


class TestCriticNode:
    @patch("lit_review_agent.graph.critique_corpus")
    def test_increments_iteration(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="approve", gaps=[], suggested_queries=[], reasoning="OK"
        )
        state = _base_state(iteration=1)
        result = critic_node(state)
        assert result["iteration"] == 2

    @patch("lit_review_agent.graph.critique_corpus")
    def test_adds_feedback(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="approve",
            gaps=[],
            suggested_queries=[],
            reasoning="Coverage adequate",
        )
        state = _base_state(critic_feedback=["prev feedback"])
        result = critic_node(state)
        assert len(result["critic_feedback"]) == 2
        assert "Coverage adequate" in result["critic_feedback"][-1]

    @patch("lit_review_agent.graph.critique_corpus")
    def test_refine_adds_queries(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="refine",
            gaps=["No sleep studies"],
            suggested_queries=["sleep staging wearable"],
            reasoning="Refine needed",
        )
        state = _base_state()
        result = critic_node(state)
        assert result["search_queries"] == ["sleep staging wearable"]


# ---------------------------------------------------------------------------
# critic_router
# ---------------------------------------------------------------------------


class TestCriticRouter:
    def test_routes_to_report_on_approve(self):
        state = _base_state(
            iteration=1,
            critic_feedback=["Iteration 0: Approved"],
        )
        assert critic_router(state) == "report"

    def test_routes_to_search_on_refine(self):
        state = _base_state(
            iteration=1,
            max_iterations=3,
            search_queries=["new query"],
            critic_feedback=["Iteration 0: Refine needed"],
        )
        assert critic_router(state) == "search"

    def test_routes_to_report_at_max_iterations(self):
        state = _base_state(
            iteration=3,
            max_iterations=3,
            search_queries=["query"],
            critic_feedback=["Iteration 2: Refine needed"],
        )
        assert critic_router(state) == "report"

    def test_routes_to_report_with_no_feedback(self):
        state = _base_state(iteration=1)
        assert critic_router(state) == "report"


# ---------------------------------------------------------------------------
# report_critic_node
# ---------------------------------------------------------------------------


class TestReportCriticNode:
    @patch("lit_review_agent.graph.critique_report")
    def test_increments_report_iteration(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="approve", issues=[], reasoning="Report is grounded."
        )
        state = _base_state(
            final_report="# Report\nSome content.",
            report_iteration=0,
        )
        result = report_critic_node(state)
        assert result["report_iteration"] == 1

    @patch("lit_review_agent.graph.critique_report")
    def test_adds_feedback(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="approve", issues=[], reasoning="All claims verified."
        )
        state = _base_state(
            final_report="# Report",
            report_critic_feedback=["prev feedback"],
        )
        result = report_critic_node(state)
        assert len(result["report_critic_feedback"]) == 2
        assert "All claims verified" in result["report_critic_feedback"][-1]

    @patch("lit_review_agent.graph.critique_report")
    def test_revise_preserves_issues(self, mock_critique):
        mock_critique.return_value = MagicMock(
            decision="revise",
            issues=["Cohort size mismatch for Paper 3"],
            reasoning="Revise needed — factual errors found.",
        )
        state = _base_state(final_report="# Report")
        result = report_critic_node(state)
        assert "Revise needed" in result["report_critic_feedback"][-1]


# ---------------------------------------------------------------------------
# report_critic_router
# ---------------------------------------------------------------------------


class TestReportCriticRouter:
    def test_routes_to_end_on_approve(self):
        state = _base_state(
            report_iteration=1,
            report_critic_feedback=["Report iteration 0: Approved"],
        )
        assert report_critic_router(state) == "end"

    def test_routes_to_report_on_revise(self):
        state = _base_state(
            report_iteration=1,
            max_report_iterations=2,
            report_critic_feedback=["Report iteration 0: Revise needed"],
        )
        assert report_critic_router(state) == "report"

    def test_routes_to_end_at_max_iterations(self):
        state = _base_state(
            report_iteration=2,
            max_report_iterations=2,
            report_critic_feedback=["Report iteration 1: Revise needed"],
        )
        assert report_critic_router(state) == "end"

    def test_routes_to_end_with_no_feedback(self):
        state = _base_state(report_iteration=1)
        assert report_critic_router(state) == "end"
