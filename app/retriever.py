from app import memory_system


# =========================================================
# PERSONAL RETRIEVAL
# =========================================================
def retrieve_personal(question, top_k=5):
    """
    Personal retrieval adapter.

    Filtering and scoring logic are handled exclusively
    inside memory_system. This function only formats results.
    """

    if not question or not question.strip():
        return []

    # Minimal index guard
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


# =========================================================
# ACADEMIC RETRIEVAL
# =========================================================
def retrieve_academic(question, top_k=5):
    """
    Academic retrieval adapter.

    All filtering, thresholding, and ranking logic
    is implemented inside memory_system.
    """

    if not question or not question.strip():
        return []

    hits = memory_system.search_academic(
        question,
        top_k=top_k,
        return_scores=True
    ) or []

    if not hits:
        return []

    return [
        {
            "text": entry.get("text"),
            "source": entry.get("source", "unknown"),
            "year": entry.get("year", "?"),
            "score": score
        }
        for entry, score in hits
    ]
