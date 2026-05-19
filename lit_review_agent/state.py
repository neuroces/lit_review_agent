"""State schema for the literature review graph."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel


class Paper(BaseModel):
    """A single paper with metadata and synthesis fields."""

    source: Literal["pubmed", "semantic_scholar"]
    doi: str | None = None
    pmid: str | None = None
    title: str
    authors: list[str] = []
    year: int
    abstract: str

    # Filled by Synthesis Agent
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


class ReviewState(TypedDict):
    """Shared state flowing through the LangGraph graph."""

    topic: str
    search_queries: list[str]
    papers: list[Paper]
    critic_feedback: list[str]
    iteration: int
    max_iterations: int
    final_report: str | None
