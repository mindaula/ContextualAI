"""Route-scoped retrieval context assembly for core orchestration.

Architectural role:
    Converts `(question, intent, confidence)` into a normalized retrieval payload used
    by `app.core.engine`. The module enforces strict route isolation so one route does
    not leak context from other memory domains.

Retrieval strategy:
    - `academic`: semantic academic retrieval via `retrieve_academic`.
    - `personal_query`: semantic personal retrieval via `retrieve_personal`, with a
      deterministic fallback exposing stored personal facts when semantic search misses.
    - `conversation_query`: hybrid merge of long-term semantic conversation hits and
      short-term recent messages from session memory.

Ranking and scoring:
    This module does not compute semantic scores itself. Numeric ranking is delegated
    to memory-layer search functions. Here, ranking is limited to:
    - fixed caps (`academic<=5`, `personal<=5`, `conversation<=6`),
    - merge order preservation for conversation context,
    - duplicate removal with first-hit retention.

FAISS interaction:
    No direct FAISS calls. Any vector index access is indirect through memory modules.

Determinism and performance:
    Deterministic for fixed inputs and memory state. Runtime is bounded by small
    retrieval caps and short-term message window limits.
"""

from app.retrieval.retriever import retrieve_personal, retrieve_academic
from app.memory.conversation_memory import retrieve_conversation
import app.memory.conversation_manager as conversation_manager


def build_memory_context(question: str, intent: str, confidence: float, domain=None):
    """Build route-isolated retrieval context for one user request.

    Args:
        question: User query after rewrite/routing preprocessing.
        intent: Route intent selected by the core routing layer.
        confidence: Router confidence score propagated to output.
        domain: Optional academic domain passed to academic retrieval.

    Returns:
        Dict with keys:
        - `route`: effective route (may be overridden for `/memory ...`),
        - `confidence`: rounded route confidence,
        - `personal`: personal retrieval hits,
        - `academic`: academic retrieval hits,
        - `conversation`: conversation retrieval snippets.

    Retrieval control flow:
        - Empty/blank questions return an empty-scaffold payload.
        - `/memory ...` force-routes to `conversation_query`.
        - Per-intent retrieval remains isolated; no cross-route blending.

    Side effects:
        Reads memory state through retrieval adapters and short-term session manager.
        Does not write persistence artifacts.

    Determinism and edge cases:
        - Deterministic for fixed memory/query state.
        - Personal fallback injects all stored personal facts (score fixed at `1.0`)
          when semantic search yields no hits and personal index is non-empty.
        - `personal_store` and unknown intents intentionally return empty context.
    """

    result = {
        "route": intent,
        "confidence": round(confidence, 4),
        "personal": [],
        "academic": [],
        "conversation": []
    }

    if not question or not question.strip():
        return result

    if question.strip().startswith("/memory"):
        cleaned_question = question.strip()[len("/memory"):].strip()
        result["route"] = "conversation_query"
        result["conversation"] = retrieve_conversation(cleaned_question, top_k=3) or []
        return result

    if intent == "academic":
        academic_hits = retrieve_academic(question, domain=domain) or []
        result["academic"] = academic_hits[:5]

    elif intent == "personal_query":
        personal_hits = retrieve_personal(question) or []

        if not personal_hits:
            import app.memory.memory_system as memory_system

            if getattr(memory_system.personal_index, "ntotal", 0) > 0:
                personal_hits = [
                    {"text": meta.get("text", ""), "score": 1.0}
                    for meta in memory_system.personal_meta
                    if meta.get("text")
                ]

        result["personal"] = personal_hits[:5]

    elif intent == "conversation_query":
        semantic_hits = retrieve_conversation(question, top_k=3) or []
        recent = conversation_manager.get_recent_messages(limit=4) or []

        formatted_recent = []

        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    formatted_recent.append(f"{role}: {content}")

        merged = semantic_hits + formatted_recent

        seen = set()
        cleaned = []

        for item in merged:
            if item and item not in seen:
                seen.add(item)
                cleaned.append(item)

        result["conversation"] = cleaned[:6]

    elif intent == "personal_store":
        pass

    else:
        pass

    return result
