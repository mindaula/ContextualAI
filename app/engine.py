import sys

from app.prompt_builder import (
    build_personal_prompt,
    build_academic_prompt,
    build_general_prompt
)

from app.safety import is_allowed
from app.context_builder import build_memory_context
from app.query_rewriter import rewrite_query
from app.llm_router import decide_route

from app import conversation_manager
from app.llm_interface import generate_answer

from app.memory_system import add_personal_fact


# =========================================================
# Duplicate Response Guard
# =========================================================

def deduplicate_response(text: str):
    """
    Attempts to remove duplicated content from model outputs.

    Applies multiple heuristics:
    - Detects full text duplication (first half == second half)
    - Removes duplicate paragraphs
    - Removes duplicate sentences
    """

    if not text:
        return ""

    text = text.strip()

    half = len(text) // 2
    if half > 20:
        first = text[:half].strip()
        second = text[half:].strip()
        if first == second:
            return first

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    unique_paragraphs = []

    for p in paragraphs:
        if p not in unique_paragraphs:
            unique_paragraphs.append(p)

    if len(unique_paragraphs) < len(paragraphs):
        return "\n\n".join(unique_paragraphs)

    sentences = text.split(". ")
    unique_sentences = []

    for s in sentences:
        s = s.strip()
        if s and s not in unique_sentences:
            unique_sentences.append(s)

    return ". ".join(unique_sentences).strip()


# =========================================================
# Safe Generation Wrapper (Streaming-Aware)
# =========================================================

def safe_generate(prompt: str, stream=True):
    """
    Wraps the LLM generation call with safety and error handling.

    - Returns a generator directly if streaming is enabled.
    - Applies deduplication only in non-streaming mode.
    """

    if not prompt or not prompt.strip():
        return "Error: Empty prompt."

    try:
        response = generate_answer(prompt, stream=stream)

    except KeyboardInterrupt:
        return ""

    except Exception as e:
        return f"LLM error: {e}"

    # If streaming is enabled, return the generator directly
    if stream:
        return response

    # Only post-process when streaming is disabled
    return deduplicate_response(response or "").strip()


# =========================================================
# Academic Evidence Prioritization
# =========================================================

def prioritize_academic_hits(academic_hits):
    """
    Sorts academic retrieval results by relevance score and publication year.
    Returns the top-ranked results.
    """

    if not academic_hits:
        return []

    def sort_key(item):
        score = item.get("score", 0)
        year = item.get("year", 0)

        try:
            year = int(year)
        except Exception:
            year = 0

        return (score, year)

    sorted_hits = sorted(
        academic_hits,
        key=sort_key,
        reverse=True
    )

    return sorted_hits[:5]


# =========================================================
# Core Engine Function
# =========================================================

def process_message(question: str):
    """
    Main orchestration function.

    Pipeline:
    1. Safety check
    2. Query rewriting
    3. Intent routing
    4. Memory context retrieval
    5. Prompt construction
    6. LLM generation
    7. Conversation persistence
    """

    if not question:
        return ""

    if not is_allowed(question):
        return "This request violates safety policies."

    try:
        last_route = conversation_manager.get_last_route()
    except Exception:
        last_route = None

    rewritten_question = rewrite_query(
        question,
        conversation_manager,
        last_route=last_route
    )

    intent, confidence = decide_route(
        rewritten_question,
        return_confidence=True,
        last_route=last_route
    )

    memory_context = build_memory_context(
        rewritten_question,
        intent,
        confidence
    )

    route = memory_context.get("route", "general")
    confidence = memory_context.get("confidence", 0.0)

    personal_hits = memory_context.get("personal", [])
    academic_hits = memory_context.get("academic", [])
    conversation_hits = memory_context.get("conversation", [])

    # =====================================================
    # Academic Route
    # =====================================================

    if route == "academic":

        if academic_hits:

            prioritized_hits = prioritize_academic_hits(academic_hits)

            context_blocks = []

            for item in prioritized_hits:
                text = item.get("text", "")
                source = item.get("source", "Unknown")
                year = item.get("year", "?")

                block = f"Source: {source} ({year})\n{text}"
                context_blocks.append(block)

            prompt = (
                "Answer the following question strictly "
                "based on the provided sources.\n"
                "Summarize the information in a structured manner.\n\n"
                f"Question:\n{rewritten_question}\n\n"
                "Sources:\n\n"
                + "\n\n---\n\n".join(context_blocks)
            )

            response = safe_generate(prompt, stream=True)

        else:
            prompt = build_general_prompt(rewritten_question)
            response = safe_generate(prompt, stream=True)

        conversation_manager.add_message("user", question)
        conversation_manager.add_message("assistant", response)
        conversation_manager.set_last_route(route)

        return response


    # =====================================================
    # Personal Fact Storage
    # =====================================================

    elif route == "personal_store":

        saved = add_personal_fact(question)
        response = (
            "Stored successfully."
            if saved
            else "Already stored or invalid input."
        )

        conversation_manager.add_message("user", question)
        conversation_manager.add_message("assistant", response)
        conversation_manager.set_last_route(route)

        return response


    # =====================================================
    # Personal Query
    # =====================================================

    elif route == "personal_query":

        if personal_hits:
            prompt = build_personal_prompt(rewritten_question, personal_hits)
            response = safe_generate(prompt, stream=True)
        else:
            response = (
                "No relevant personal information found."
            )

        conversation_manager.add_message("user", question)
        conversation_manager.add_message("assistant", response)
        conversation_manager.set_last_route(route)

        return response


    # =====================================================
    # Conversation Query
    # =====================================================

    elif route == "conversation_query":

        clean_hits = [h for h in conversation_hits if h and h.strip()]

        if clean_hits:

            context_text = "\n\n---\n\n".join(clean_hits)

            prompt = (
                "Answer the question strictly based on "
                "the provided conversation history.\n\n"
                "If the information is not present, clearly "
                "state that it was not mentioned.\n\n"
                f"Question:\n{rewritten_question}\n\n"
                f"Conversation History:\n{context_text}\n\n"
                "Answer:"
            )

            response = safe_generate(prompt, stream=True)
        else:
            response = (
                "No relevant information found in previous conversations."
            )

        conversation_manager.add_message("user", question)
        conversation_manager.add_message("assistant", response)
        conversation_manager.set_last_route(route)

        return response


    # =====================================================
    # General Route
    # =====================================================

    else:

        prompt = build_general_prompt(rewritten_question)
        response = safe_generate(prompt, stream=True)

        conversation_manager.add_message("user", question)
        conversation_manager.add_message("assistant", response)
        conversation_manager.set_last_route(route)

        return response
