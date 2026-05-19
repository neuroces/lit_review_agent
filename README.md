# Multi-Agent Literature Review System for Health AI Evaluation

A multi-agent system that takes a Health AI evaluation topic and produces a structured
markdown literature review with a methodology comparison table. Built with
[LangGraph](https://github.com/langchain-ai/langgraph) and the Anthropic API.

## Why Not Just Ask ChatGPT?

Off-the-shelf LLMs can produce fluent literature reviews, but they have fundamental limitations
that this system is designed to address:

| Problem with a single-prompt LLM review | How this system solves it |
|------------------------------------------|--------------------------|
| **Hallucinated citations** — LLMs frequently fabricate paper titles, authors, and DOIs that look plausible but don't exist | Every paper comes from a verified PubMed or Semantic Scholar API call. The corpus is real and traceable. |
| **Frozen knowledge** — LLMs have a training cutoff and can't access papers published after it | Searches are live against current databases, so the review always reflects the latest literature. |
| **No structured extraction** — a single prompt produces prose, not queryable data | Each paper is decomposed into structured fields (devices, cohort size, metrics, split strategy) via Pydantic-validated extraction, producing both a readable report and a machine-readable dataset. |
| **Shallow coverage** — a single prompt can only reference papers the LLM memorized during training, biased toward popular results | Systematic database queries with deduplication surface papers an LLM would never recall, including niche or recent work. |
| **No self-correction** — if the review has gaps, you have to manually notice and re-prompt | The Critic agent automatically identifies coverage gaps (missing device types, narrow demographics, underrepresented tasks) and loops back with targeted queries — no human intervention needed. |
| **Not reproducible** — re-asking the same question produces different results | The search queries, API responses, and structured extractions are deterministic and auditable. The critic feedback trail documents every iteration. |
| **No methodology comparison** — prose reviews bury methodological differences in paragraphs | The Report agent produces a structured comparison table (devices, cohort sizes, metrics, split strategies) that makes cross-study comparison immediate. |

In short: this system treats literature review as a **data pipeline**, not a creative writing task.
The LLM is used where it excels (extracting structured information from abstracts, evaluating
coverage, generating prose) while the retrieval, deduplication, and quality control are handled
by deterministic code.

## How It Works

You provide a research topic (e.g., *"Wearable PPG-based atrial fibrillation detection"*)
and optional search queries. The system runs a pipeline of 4 AI agents that:

1. **Search** — queries PubMed and Semantic Scholar for relevant papers
2. **Synthesize** — uses Claude to extract structured fields from each abstract (task, devices, cohort, metrics, etc.)
3. **Critique** — evaluates corpus coverage and decides whether more papers are needed
4. **Report** — generates a full markdown literature review with comparison tables

The Critic agent creates a feedback loop: if it finds coverage gaps (e.g., missing device types,
narrow demographics, or underrepresented task families), it sends new search queries back to the
Search agent. This loop repeats until the Critic approves the corpus or `max_iterations` is reached.

## Architecture

```mermaid
graph TD
    START([Start]) --> Search
    Search --> Synthesize
    Synthesize --> Critic
    Critic -->|gaps found & iter < max| Search
    Critic -->|approved or max iter| Report
    Report --> END([End])
```

### Agent Details

| Agent | Module | Role | Input → Output |
|-------|--------|------|-----------------|
| **Search** | `tools.py` | Query PubMed + Semantic Scholar, deduplicate | `search_queries` → `papers` |
| **Synthesis** | `synthesis.py` | Extract structured fields from abstracts via Claude | `papers` (raw) → `papers` (with synthesis fields) |
| **Critic** | `critic.py` | Evaluate coverage, decide refine vs. approve | `papers` + `topic` → `CriticDecision` |
| **Report** | `report.py` | Generate markdown literature review | `papers` + `topic` → `final_report` |

### Shared State

All agents communicate through a `ReviewState` dictionary:

```python
class ReviewState(TypedDict):
    topic: str                    # Research topic
    search_queries: list[str]     # Current search queries
    papers: list[Paper]           # Accumulated corpus
    critic_feedback: list[str]    # Audit trail of critic decisions
    iteration: int                # Current iteration count
    max_iterations: int           # Max allowed loops
    final_report: str | None      # Output markdown
```

Each `Paper` carries both bibliographic metadata (title, authors, year, abstract, DOI) and
synthesis fields extracted by Claude (task, devices, sensor modalities, cohort size,
evaluation metrics, key findings, limitations, etc.).

### Key Design Decisions

- **Incremental synthesis** — only unsynthesized papers are sent to Claude on each iteration,
  so previously processed papers aren't re-processed during loop-back
- **Bounded loops** — `max_iterations` (default 3) prevents infinite critic loops
- **Graceful degradation** — search functions return `[]` on persistent rate limits,
  Critic defaults to "approve" on parse failures (prevents getting stuck)
- **Prompt caching** — system prompts use Anthropic's `cache_control` to save ~90% on input
  tokens for the 2nd+ paper in each batch
- **Configurable model** — set `ANTHROPIC_MODEL` in `secrets.txt` to use a different Claude model

## Project Structure

```
lit_review_agent/
├── lit_review_agent/
│   ├── __init__.py
│   ├── config.py          # API key loading, client factories
│   ├── state.py           # Paper model + ReviewState schema
│   ├── tools.py           # PubMed/S2 search + deduplication
│   ├── synthesis.py       # Synthesis Agent (Claude extraction)
│   ├── critic.py          # Critic Agent (coverage evaluation)
│   ├── report.py          # Report Agent (markdown generation)
│   └── graph.py           # LangGraph wiring + run_review()
├── tests/
│   ├── test_tools_and_synthesis.py  # 23 unit tests (mocked)
│   ├── test_critic_and_report.py    # 13 unit tests (mocked)
│   ├── test_graph.py                # 13 unit tests (mocked)
│   └── test_integration.py          # 6 integration tests (real APIs)
├── notebooks/
│   ├── test_tools.ipynb             # Interactive search testing
│   ├── test_synthesis.ipynb         # Synthesis testing
│   ├── tutorial_synthesis.ipynb     # Synthesis agent walkthrough
│   └── tutorial_langgraph.ipynb     # LangGraph wiring walkthrough
├── pyproject.toml
├── secrets.txt            # API keys (git-ignored)
└── README.md
```

## Setup

```bash
# Clone and install
git clone <repo-url> && cd lit_review_agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure API keys — edit secrets.txt (git-ignored)
# Only ANTHROPIC_API_KEY is required
```

### secrets.txt

```
ANTHROPIC_API_KEY=sk-ant-...
# Optional: override default model
# ANTHROPIC_MODEL=claude-sonnet-4-6
# Optional: higher S2 rate limits (requires academic email)
# SEMANTIC_SCHOLAR_API_KEY=
# Optional: higher PubMed rate limits
# NCBI_API_KEY=
```

## Usage

### Python API

```python
from lit_review_agent.graph import run_review

result = run_review(
    topic="Wearable device validation for atrial fibrillation detection",
    initial_queries=[
        "smartwatch PPG atrial fibrillation deep learning",
        "wearable ECG AFib validation benchmark",
    ],
    max_iterations=2,
)

print(f"Papers: {len(result['papers'])}")
print(f"Iterations: {result['iteration']}")
print(result["final_report"])
```

### Running Tests

```bash
# Unit tests (fast, mocked, no API key needed)
pytest tests/ -v

# Integration tests (real APIs, ~5 min, requires API key)
pytest tests/test_integration.py -v -m integration
```

## License

MIT
