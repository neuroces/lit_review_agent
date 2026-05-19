"""LangGraph graph wiring — orchestrates Search → Synthesize → Critic → Report.

The graph implements a feedback loop:

    START → search_node → synthesize_node → critic_node ─┐
                ↑                                         │
                └──── (refine: add queries, loop back) ───┘
                                                          │
                                        (approve) → report_node → END

The Critic acts as a router: if coverage gaps exist and iterations remain,
it sends the graph back to search with new queries. Otherwise, it proceeds
to report generation.
"""

from __future__ import annotations

import logging
import time

from langgraph.graph import END, START, StateGraph
from lit_review_agent.critic import critique_corpus
from lit_review_agent.report import generate_report
from lit_review_agent.state import Paper, ReviewState
from lit_review_agent.synthesis import synthesize_papers
from lit_review_agent.tools import dedupe_papers, search_pubmed, search_semantic_scholar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions — each takes ReviewState, returns partial state update
# ---------------------------------------------------------------------------


def search_node(state: ReviewState) -> dict:
    """Search PubMed and Semantic Scholar for papers matching current queries."""
    queries = state["search_queries"]
    existing_papers = state.get("papers", [])

    logger.info(
        "Search node: %d queries, %d existing papers",
        len(queries),
        len(existing_papers),
    )

    new_papers: list[Paper] = []
    for q in queries:
        pm_results = search_pubmed(q, max_results=15)
        new_papers.extend(pm_results)
        # Brief pause between PubMed and S2
        time.sleep(0.5)
        s2_results = search_semantic_scholar(q, max_results=15)
        new_papers.extend(s2_results)

    # Combine with existing and deduplicate
    all_papers = existing_papers + new_papers
    unique = dedupe_papers(all_papers)

    logger.info(
        "Search node: fetched %d new, total %d unique (was %d)",
        len(new_papers),
        len(unique),
        len(existing_papers),
    )

    return {"papers": unique}


def synthesize_node(state: ReviewState) -> dict:
    """Extract structured fields from all unsynthesized papers."""
    papers = state["papers"]

    # Only synthesize papers that haven't been processed yet (task is None)
    unsynthesized = [p for p in papers if p.task is None]
    already_done = [p for p in papers if p.task is not None]

    logger.info(
        "Synthesize node: %d to process, %d already done",
        len(unsynthesized),
        len(already_done),
    )

    if unsynthesized:
        newly_synthesized = synthesize_papers(unsynthesized)
        all_papers = already_done + newly_synthesized
    else:
        all_papers = papers

    return {"papers": all_papers}


def critic_node(state: ReviewState) -> dict:
    """Review corpus coverage and decide whether to refine or approve."""
    topic = state["topic"]
    papers = state["papers"]
    iteration = state.get("iteration", 0)

    decision = critique_corpus(topic, papers)

    logger.info(
        "Critic node (iter %d): decision=%s, %d gaps",
        iteration,
        decision.decision,
        len(decision.gaps),
    )

    feedback = state.get("critic_feedback", [])
    if decision.reasoning:
        feedback = feedback + [f"Iteration {iteration}: {decision.reasoning}"]

    update: dict = {
        "critic_feedback": feedback,
        "iteration": iteration + 1,
    }

    # If refining, add the suggested queries for the next search round
    if decision.decision == "refine":
        update["search_queries"] = decision.suggested_queries

    return update


def report_node(state: ReviewState) -> dict:
    """Generate the final markdown literature review."""
    topic = state["topic"]
    papers = state["papers"]

    logger.info("Report node: generating review from %d papers", len(papers))

    report = generate_report(topic, papers)
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Conditional edge: critic decides next step
# ---------------------------------------------------------------------------


def critic_router(state: ReviewState) -> str:
    """Route after critic: loop back to search or proceed to report."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # Check if the last critic feedback indicates refinement
    feedback = state.get("critic_feedback", [])
    last_feedback = feedback[-1] if feedback else ""

    # If we've hit max iterations, always proceed to report
    if iteration >= max_iter:
        logger.info("Critic router: max iterations (%d) reached → report", max_iter)
        return "report"

    # Check if critic requested refinement (look at whether new queries were added)
    queries = state.get("search_queries", [])
    if queries and "refine" in last_feedback.lower():
        logger.info("Critic router: refinement requested → search")
        return "search"

    logger.info("Critic router: approved → report")
    return "report"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and return the (uncompiled) LangGraph StateGraph."""
    graph = StateGraph(ReviewState)

    # Add nodes
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report", report_node)

    # Linear edges
    graph.add_edge(START, "search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", "critic")

    # Conditional edge from critic
    graph.add_conditional_edges(
        "critic",
        critic_router,
        {"search": "search", "report": "report"},
    )

    # Report → END
    graph.add_edge("report", END)

    return graph


def compile_graph():
    """Build and compile the graph, ready to invoke."""
    return build_graph().compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_review(
    topic: str,
    initial_queries: list[str] | None = None,
    max_iterations: int = 3,
) -> ReviewState:
    """Run the full literature review pipeline.

    Args:
        topic: The Health AI topic to review.
        initial_queries: Search queries to start with. If None, uses the topic directly.
        max_iterations: Max critic feedback loops before forcing report generation.

    Returns:
        Final ReviewState with papers and report.
    """
    if initial_queries is None:
        initial_queries = [topic]

    initial_state: ReviewState = {
        "topic": topic,
        "search_queries": initial_queries,
        "papers": [],
        "critic_feedback": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_report": None,
    }

    app = compile_graph()
    final_state = app.invoke(initial_state)
    return final_state
