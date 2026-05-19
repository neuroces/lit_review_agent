"""Unit tests for tools and synthesis modules.

All external API calls are mocked — tests run offline and fast.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lit_review_agent.state import Paper
from lit_review_agent.synthesis import SynthesisExtraction, synthesize_paper
from lit_review_agent.tools import (
    _extract_doi_from_medline,
    _normalize_title,
    dedupe_papers,
    search_pubmed,
    search_semantic_scholar,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MEDLINE = """\
PMID- 12345678
TI  - Deep Learning for Atrial Fibrillation Detection Using Smartwatch PPG
AU  - Smith J
AU  - Doe A
DP  - 2024 Mar
AB  - We developed a deep learning model for AFib detection using PPG from Apple Watch.
AID - 10.1234/test.2024.001 [doi]

PMID- 87654321
TI  - A Second Paper Without Abstract

PMID- 11111111
TI  - Wearable Activity Recognition Benchmark
AU  - Lee K
DP  - 2023
AB  - A benchmark for human activity recognition using wrist-worn accelerometers.
AID - 10.5678/har.2023 [doi]
"""


def _make_paper(
    title: str = "Test Paper",
    doi: str | None = "10.1234/test",
    source: str = "pubmed",
    year: int = 2024,
    abstract: str = "Test abstract about wearable AFib detection using PPG.",
) -> Paper:
    return Paper(
        source=source,
        doi=doi,
        pmid="12345",
        title=title,
        authors=["Smith J"],
        year=year,
        abstract=abstract,
    )


# ---------------------------------------------------------------------------
# Tools: _extract_doi_from_medline
# ---------------------------------------------------------------------------


class TestExtractDoi:
    def test_extracts_doi(self):
        rec = {"AID": ["10.1234/test [doi]", "PMC12345 [pmc]"]}
        assert _extract_doi_from_medline(rec) == "10.1234/test"

    def test_no_doi_returns_none(self):
        rec = {"AID": ["PMC12345 [pmc]"]}
        assert _extract_doi_from_medline(rec) is None

    def test_empty_aid_returns_none(self):
        assert _extract_doi_from_medline({}) is None


# ---------------------------------------------------------------------------
# Tools: _normalize_title
# ---------------------------------------------------------------------------


class TestNormalizeTitle:
    def test_lowercases(self):
        assert _normalize_title("Deep Learning") == "deep learning"

    def test_strips_punctuation(self):
        assert _normalize_title("AFib: A Study!") == "afib a study"

    def test_handles_empty(self):
        assert _normalize_title("") == ""

    def test_equivalent_titles_match(self):
        t1 = "Deep Learning for AFib Detection"
        t2 = "Deep learning for AFib detection."
        assert _normalize_title(t1) == _normalize_title(t2)


# ---------------------------------------------------------------------------
# Tools: dedupe_papers
# ---------------------------------------------------------------------------


class TestDedupePapers:
    def test_dedupes_by_doi(self):
        p1 = _make_paper(title="Paper A", doi="10.1234/same")
        p2 = _make_paper(title="Paper B", doi="10.1234/same")
        result = dedupe_papers([p1, p2])
        assert len(result) == 1
        assert result[0].title == "Paper A"

    def test_doi_case_insensitive(self):
        p1 = _make_paper(title="Paper A", doi="10.1234/ABC")
        p2 = _make_paper(title="Paper B", doi="10.1234/abc")
        assert len(dedupe_papers([p1, p2])) == 1

    def test_dedupes_by_title_when_no_doi(self):
        p1 = _make_paper(title="Same Title", doi=None)
        p2 = _make_paper(title="Same Title!", doi=None)  # punctuation differs
        assert len(dedupe_papers([p1, p2])) == 1

    def test_keeps_different_papers(self):
        p1 = _make_paper(title="Paper A", doi="10.1/a")
        p2 = _make_paper(title="Paper B", doi="10.1/b")
        assert len(dedupe_papers([p1, p2])) == 2

    def test_empty_input(self):
        assert dedupe_papers([]) == []

    def test_preserves_order(self):
        papers = [_make_paper(title=f"Paper {i}", doi=f"10.1/{i}") for i in range(5)]
        result = dedupe_papers(papers)
        assert [p.title for p in result] == [f"Paper {i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Tools: search_pubmed (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSearchPubmed:
    @patch("lit_review_agent.tools.time.sleep")  # skip delays
    @patch("lit_review_agent.tools.get_ncbi_api_key", return_value=None)
    def test_returns_papers(self, _mock_key, _mock_sleep):
        esearch_resp = httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["12345678", "11111111"]}},
            request=httpx.Request("GET", "http://test"),
        )
        efetch_resp = httpx.Response(
            200,
            text=SAMPLE_MEDLINE,
            request=httpx.Request("GET", "http://test"),
        )

        with patch("lit_review_agent.tools.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = [esearch_resp, efetch_resp]

            papers = search_pubmed("test query", max_results=5)

        # Should skip the paper without abstract
        assert len(papers) == 2
        assert papers[0].pmid == "12345678"
        assert papers[0].doi == "10.1234/test.2024.001"
        assert papers[0].year == 2024
        assert papers[0].source == "pubmed"
        assert papers[1].pmid == "11111111"

    @patch("lit_review_agent.tools.time.sleep")
    @patch("lit_review_agent.tools.get_ncbi_api_key", return_value=None)
    def test_empty_results(self, _mock_key, _mock_sleep):
        esearch_resp = httpx.Response(
            200,
            json={"esearchresult": {"idlist": []}},
            request=httpx.Request("GET", "http://test"),
        )

        with patch("lit_review_agent.tools.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = esearch_resp

            papers = search_pubmed("nonexistent topic", max_results=5)

        assert papers == []


# ---------------------------------------------------------------------------
# Tools: search_semantic_scholar (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSearchSemanticScholar:
    @patch("lit_review_agent.tools.time.sleep")
    @patch("lit_review_agent.tools.get_semantic_scholar_headers", return_value={})
    def test_returns_papers(self, _mock_headers, _mock_sleep):
        s2_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "externalIds": {"DOI": "10.1/s2test", "PubMed": "99999"},
                    "title": "S2 Test Paper",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "year": 2023,
                    "abstract": "A study on wearables.",
                },
                {
                    "paperId": "def456",
                    "externalIds": {},
                    "title": "No Abstract Paper",
                    "authors": [],
                    "year": 2023,
                    "abstract": None,  # should be skipped
                },
            ]
        }
        resp = httpx.Response(
            200, json=s2_data, request=httpx.Request("GET", "http://test")
        )

        with patch("lit_review_agent.tools.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = resp

            papers = search_semantic_scholar("test", max_results=5)

        assert len(papers) == 1
        assert papers[0].source == "semantic_scholar"
        assert papers[0].doi == "10.1/s2test"
        assert papers[0].authors == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# Synthesis: SynthesisExtraction schema
# ---------------------------------------------------------------------------


class TestSynthesisExtraction:
    def test_valid_full_extraction(self):
        data = {
            "task": "AFib detection",
            "devices": ["Apple Watch"],
            "sensor_modalities": ["PPG"],
            "cohort_size": 100,
            "cohort_demographics": "ages 40-80",
            "reference_standard": "12-lead ECG",
            "split_strategy": "participant-level",
            "evaluation_metrics": ["sensitivity", "specificity"],
            "key_findings": "Model achieved 95% sensitivity.",
            "limitations": "Small cohort",
            "quality_notes": "External validation included",
        }
        ext = SynthesisExtraction(**data)
        assert ext.task == "AFib detection"
        assert ext.cohort_size == 100

    def test_minimal_extraction(self):
        ext = SynthesisExtraction()
        assert ext.task is None
        assert ext.devices == []
        assert ext.evaluation_metrics == []

    def test_null_fields_accepted(self):
        data = {"task": None, "cohort_size": None, "devices": []}
        ext = SynthesisExtraction(**data)
        assert ext.task is None


# ---------------------------------------------------------------------------
# Synthesis: synthesize_paper (mocked Anthropic client)
# ---------------------------------------------------------------------------


class TestSynthesizePaper:
    def _mock_response(self, json_text: str) -> MagicMock:
        """Create a mock Anthropic API response."""
        mock = MagicMock()
        mock.content = [MagicMock(text=json_text)]
        mock.usage = MagicMock(
            input_tokens=500,
            output_tokens=200,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=0,
        )
        return mock

    @patch("lit_review_agent.synthesis.get_anthropic_client")
    def test_successful_extraction(self, mock_get_client):
        extraction_json = json.dumps(
            {
                "task": "AFib detection",
                "devices": ["Apple Watch"],
                "sensor_modalities": ["PPG"],
                "cohort_size": 50,
                "cohort_demographics": None,
                "reference_standard": "12-lead ECG",
                "split_strategy": "participant-level",
                "evaluation_metrics": ["sensitivity", "AUROC"],
                "key_findings": "Model achieved 95% sensitivity.",
                "limitations": None,
                "quality_notes": "Well-designed study",
            }
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(extraction_json)
        mock_get_client.return_value = mock_client

        paper = _make_paper()
        result = synthesize_paper(paper)

        assert result.task == "AFib detection"
        assert result.devices == ["Apple Watch"]
        assert result.cohort_size == 50
        assert result.evaluation_metrics == ["sensitivity", "AUROC"]
        # Original fields preserved
        assert result.title == paper.title
        assert result.abstract == paper.abstract

    @patch("lit_review_agent.synthesis.get_anthropic_client")
    def test_handles_markdown_fences(self, mock_get_client):
        extraction_json = '```json\n{"task": "sleep staging", "devices": []}\n```'
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(extraction_json)
        mock_get_client.return_value = mock_client

        result = synthesize_paper(_make_paper())
        assert result.task == "sleep staging"

    @patch("lit_review_agent.synthesis.get_anthropic_client")
    def test_retries_on_bad_json(self, mock_get_client):
        bad_resp = self._mock_response("this is not json")
        good_resp = self._mock_response('{"task": "HAR", "devices": []}')

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [bad_resp, good_resp]
        mock_get_client.return_value = mock_client

        result = synthesize_paper(_make_paper())
        assert result.task == "HAR"
        assert mock_client.messages.create.call_count == 2

    @patch("lit_review_agent.synthesis.get_anthropic_client")
    def test_returns_original_on_all_retries_exhausted(self, mock_get_client):
        bad_resp = self._mock_response("not json")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = bad_resp
        mock_get_client.return_value = mock_client

        paper = _make_paper()
        result = synthesize_paper(paper, max_retries=1)

        # Should return original paper unmodified
        assert result.task is None
        assert result.title == paper.title
