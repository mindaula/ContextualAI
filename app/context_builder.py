# context_builder.py

from app.retriever import retrieve_personal, retrieve_academic
from app.conversation_memory import retrieve_conversation
from app import conversation_manager


# =========================================================
# Build Memory Context (Router-Driven, No Route Mutation)
# =========================================================

def build_memory_context(question: str, intent: str, confidence: float):
    """
    Builds a structured memory context based on the router's decision.

    The router's intent is treated as authoritative and is never modified.
    Retrieval components only provide additional contextual data.

    Args:
        question (str): The user query.
        intent (str): The classified intent determined by the router.
        confidence (float): Confidence score assigned by the router.

    Returns:
        dict: A structured context object containing:
            - route (str)
            - confidence (float)
            - personal (list)
            - academic (list)
            - conversation (list)
    """

    result = {
        "route": intent,                      # Router remains authoritative
        "confidence": round(confidence, 4),   # Confidence is preserved
        "personal": [],
        "academic": [],
        "conversation": []
    }

    if not question or not question.strip():
        return result  # No fallback logic applied here

    # =====================================================
    # Router-Driven Retrieval (No Route Override)
    # =====================================================

    # Academic intent
    if intent == "academic":
        result["academic"] = retrieve_academic(question)

    # Personal query intent
    elif intent == "personal_query":
        result["personal"] = retrieve_personal(question)

    # Conversation-related intent
    elif intent == "conversation_query":

        # Semantic retrieval (vector search)
        semantic_hits = retrieve_conversation(question, top_k=5)

        # Short-term memory (recent conversation history)
        recent = conversation_manager.get_recent_messages(limit=6)

        formatted_conversation = []

        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    formatted_conversation.append(f"{role}: {content}")

        # Merge semantic retrieval results with recent history
        merged = semantic_hits + formatted_conversation

        # Deduplicate while preserving order
        result["conversation"] = list(dict.fromkeys(merged))

    # Personal store intent (no retrieval required)
    elif intent == "personal_store":
        pass

    # General intent (no retrieval required)
    else:
        pass

    return result
