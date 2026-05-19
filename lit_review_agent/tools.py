"""Search and deduplication tools for literature retrieval."""

from __future__ import annotations

import logging
import re
import time
from io import StringIO

import httpx
from Bio import Medline
from lit_review_agent.config import get_ncbi_api_key, get_semantic_scholar_headers
from lit_review_agent.state import Paper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared HTTP helper with retry
# ---------------------------------------------------------------------------


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 5,
    label: str = "",
) -> httpx.Response:
    """GET with exponential backoff on 429 responses."""
    for attempt in range(max_retries + 1):
        resp = client.get(url, params=params, headers=headers or {})
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            logger.warning("%s rate-limited (429), retrying in %ds...", label, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp

    logger.warning(
        "%s rate limit exceeded after %d retries — returning None", label, max_retries
    )
    return resp  # caller checks status


# ---------------------------------------------------------------------------
# PubMed via NCBI E-utilities (httpx + Biopython Medline parser)
# ---------------------------------------------------------------------------

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 20) -> list[Paper]:
    """Search PubMed and return Paper objects with metadata.

    Uses NCBI E-utilities: esearch to get IDs, then efetch for metadata.
    """
    api_key = get_ncbi_api_key()
    base_params: dict[str, str] = {
        "tool": "lit_review_agent",
        "email": "lit-review-agent@example.com",
    }
    if api_key:
        base_params["api_key"] = api_key

    with httpx.Client(timeout=60) as client:
        # Step 1: search for matching PMIDs
        search_params = {
            **base_params,
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
        }
        resp = _get_with_retry(
            client, f"{_NCBI_BASE}/esearch.fcgi", params=search_params, label="PubMed"
        )
        if resp.status_code == 429:
            logger.warning("PubMed search rate-limited — returning empty results.")
            return []
        search_data = resp.json()

        pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            logger.info("PubMed: no results for query=%r", query)
            return []

        logger.info("PubMed: found %d results for query=%r", len(pmids), query)

        # Brief pause between esearch and efetch to respect rate limits
        time.sleep(0.4)

        # Step 2: fetch metadata in MEDLINE format
        fetch_params = {
            **base_params,
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "medline",
            "retmode": "text",
        }
        resp = _get_with_retry(
            client, f"{_NCBI_BASE}/efetch.fcgi", params=fetch_params, label="PubMed"
        )
        if resp.status_code == 429:
            logger.warning("PubMed fetch rate-limited — returning empty results.")
            return []

    # Parse MEDLINE records using Biopython
    records = list(Medline.parse(StringIO(resp.text)))

    papers: list[Paper] = []
    for rec in records:
        # Extract year from date field (e.g. "2023 Jan 15" or "2023")
        date_str = rec.get("DP", "")
        year_match = re.search(r"(\d{4})", date_str)
        year = int(year_match.group(1)) if year_match else 0

        abstract = rec.get("AB", "")
        if not abstract:
            continue  # skip papers without abstracts

        papers.append(
            Paper(
                source="pubmed",
                pmid=rec.get("PMID"),
                doi=_extract_doi_from_medline(rec),
                title=rec.get("TI", ""),
                authors=rec.get("AU", []),
                year=year,
                abstract=abstract,
            )
        )

    return papers


def _extract_doi_from_medline(rec: dict) -> str | None:
    """Extract DOI from MEDLINE AID field."""
    aids = rec.get("AID", [])
    for aid in aids:
        if aid.endswith("[doi]"):
            return aid.replace(" [doi]", "").strip()
    return None


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "paperId,externalIds,title,authors,year,abstract"


def search_semantic_scholar(query: str, max_results: int = 20) -> list[Paper]:
    """Search Semantic Scholar and return Paper objects.

    Without an API key, rate-limited to ~1 req/sec.
    Retries with exponential backoff on 429 responses.
    Returns empty list (with warning) if rate limit cannot be resolved.
    """
    headers = get_semantic_scholar_headers()

    params = {
        "query": query,
        "limit": min(max_results, 100),  # S2 max per request
        "fields": _S2_FIELDS,
    }

    # Throttle unauthenticated requests
    if not headers:
        time.sleep(2.0)

    with httpx.Client(timeout=30) as client:
        resp = _get_with_retry(
            client,
            f"{_S2_BASE}/paper/search",
            params=params,
            headers=headers,
            label="S2",
        )
        if resp.status_code == 429:
            logger.warning(
                "Semantic Scholar rate limit exceeded — returning empty results. "
                "Get an API key for reliable access."
            )
            return []

        data = resp.json()

    raw_papers = data.get("data", [])
    logger.info(
        "Semantic Scholar: found %d results for query=%r", len(raw_papers), query
    )

    papers: list[Paper] = []
    for p in raw_papers:
        abstract = p.get("abstract") or ""
        if not abstract:
            continue

        ext_ids = p.get("externalIds") or {}
        author_names = [a.get("name", "") for a in (p.get("authors") or [])]

        papers.append(
            Paper(
                source="semantic_scholar",
                doi=ext_ids.get("DOI"),
                pmid=ext_ids.get("PubMed"),
                title=p.get("title", ""),
                authors=author_names,
                year=p.get("year") or 0,
                abstract=abstract,
            )
        )

    return papers


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy title matching."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by DOI first, then by normalized title."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Paper] = []

    for p in papers:
        # Check DOI first (most reliable)
        if p.doi:
            doi_lower = p.doi.lower()
            if doi_lower in seen_dois:
                continue
            seen_dois.add(doi_lower)

        # Fall back to normalized title
        norm = _normalize_title(p.title)
        if norm in seen_titles:
            continue
        seen_titles.add(norm)

        unique.append(p)

    logger.info("Dedup: %d -> %d papers", len(papers), len(unique))
    return unique
