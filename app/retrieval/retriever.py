"""Thin retrieval adapters around memory-layer search functions.

Architectural role:
    Normalizes outputs from `app.memory.memory_system` into structures expected by
    `app.retrieval.context_builder` and `app.core.engine`.

Retrieval and ranking model:
    This module does not implement semantic ranking or scoring formulas. It delegates
    ranking to memory-layer FAISS-backed searches:
    - `search_personal(...)`
    - `search_academic(...)`
    Returned order and scores are preserved.

FAISS interaction:
    Indirect only. No direct FAISS API calls are made here.

Determinism and performance:
    Deterministic for fixed memory search outputs. Runtime overhead is minimal
    (input guards + lightweight output formatting).
"""

import app.memory.memory_system as memory_system


def retrieve_personal(question, top_k=5):
    """Retrieve and format personal-memory hits.

    Args:
        question: User query.
        top_k: Maximum hits requested from memory-layer search.

    Returns:
        List of dicts: `{"text": <fact>, "score": <float>}`.

    Ranking logic:
        Ranking and score thresholds are delegated to
        `memory_system.search_personal(...)`. This function preserves returned order.

    Edge cases:
        - Empty query returns `[]`.
        - Missing/empty personal index returns `[]`.
        - Missing hits return `[]`.
    """

    if not question or not question.strip():
        return []

    if hasattr(memory_system, "personal_index"):
        if getattr(memory_system.personal_index, "ntotal", 0) < 1:
            return []

    hits = memory_system.search_personal(
        question,
        top_k=top_k,
        return_scores=True
    ) or []

    if not hits:
        return []

    return [
        {
            "text": text,
            "score": score
        }
        for text, score in hits
    ]


def retrieve_academic(question, top_k=5, domain=None):
    """Retrieve and format academic-memory hits.

    Args:
        question: User query.
        top_k: Maximum hits requested from memory-layer search.
        domain: Optional domain identifier for domain-scoped academic index lookup.

    Returns:
        List of dict entries enriched with stable keys:
        `text`, `source`, `year`, and delegated `score`.

    Ranking logic and scoring formula:
        Entirely delegated to `memory_system.search_academic(...)`, including its
        thresholds/recency adjustments. This adapter does not recompute scores.

    FAISS interaction:
        Indirect through `memory_system`.

    Edge cases:
        - Empty query returns `[]`.
        - Missing hits return `[]`.
    """

    if not question or not question.strip():
        return []

    hits = memory_system.search_academic(
        question,
        top_k=top_k,
        return_scores=True,
        domain=domain
    ) or []

    if not hits:
        return []

    return [
        {
            **entry,
            "text": entry.get("text"),
            "source": entry.get("source", "unknown"),
            "year": entry.get("year", "?"),
            "score": score
        }
        for entry, score in hits
    ]
