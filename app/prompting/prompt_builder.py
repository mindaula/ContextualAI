"""Prompt assembly helpers used by core orchestration.

This module is intentionally narrow: it only builds prompt strings from already
routed inputs. Route selection, retrieval, validation, token budgeting, and model
invocation happen outside this module.

Design constraints:
    - Deterministic construction for identical inputs.
    - Fixed ordering of prompt components per route.
    - No hidden side effects (no I/O, no global state mutation).

Prompt safety model:
    - Safety is instruction-led, not parser-enforced.
    - User text and injected context are interpolated as raw strings.
    - Upstream layers are responsible for sanitization, trust boundaries, and
      context-window limits.
"""

from typing import List


# =========================================================
# SYSTEM IDENTITY (GLOBAL)
# =========================================================
# Shared prefix appended first in every prompt variant.
# This is the only global system-level instruction in this module.
# Ordering guarantee: `SYSTEM_IDENTITY` is always prepended before any route-
# specific instructions, injected context, or user question.

SYSTEM_IDENTITY = (
    "You are 'Advanced AI Tutor', a locally operated AI tutor.\n"
    "You are not affiliated with any external company or service.\n"
    "Do not claim to be a specific commercial model.\n"
    "Respond clearly, precisely, and without repetition.\n"
    "Do not repeat sentences or paragraphs.\n\n"
)


# =========================================================
# PERSONAL PROMPT
# =========================================================
# Used for memory-backed personal question answering.
# Memory injection strategy:
#   - Accepts iterable `facts` from upstream memory retrieval.
#   - Each item is normalized:
#       * dict -> `item.get("text", "").strip()`
#       * other -> `str(item).strip()`
#   - Empty values are dropped; fallback line is inserted when no usable facts.
# Prompt component order:
#   1) `SYSTEM_IDENTITY`
#   2) Personal-answering constraints
#   3) Injected memory block ("Known information")
#   4) User question
#   5) Assistant cue ("Answer:")

def build_personal_prompt(question: str, facts) -> str:
    """Build a personal-memory prompt from user question and memory facts.

    Args:
        question: Final user question text selected by upstream routing.
        facts: Memory entries (usually dicts with `text`) from retrieval.

    Returns:
        Fully assembled prompt string for model invocation.

    Determinism:
        Deterministic for identical input values and iteration order of `facts`.

    Edge cases:
        - If no usable facts remain after normalization, injects
          "No stored personal facts available." as explicit fallback context.
        - `question` is stripped before insertion.

    Failure handling:
        No local exception handling is applied. Non-iterable/invalid `facts`
        types raise upstream-visible exceptions.

    Prompt safety considerations:
        Facts are injected as plain text bullets without escaping. Safety relies
        on upstream filtering and the behavioral instructions in the prompt.
    """

    cleaned_facts = []

    for f in facts:
        if isinstance(f, dict):
            text = f.get("text", "").strip()
        else:
            text = str(f).strip()

        if text:
            cleaned_facts.append(text)

    if not cleaned_facts:
        cleaned_facts.append("No stored personal facts available.")

    context_block = "\n".join(f"- {fact}" for fact in cleaned_facts)

    return (
        SYSTEM_IDENTITY +
        "The following information about the user is known.\n"
        "Use it strictly as background knowledge.\n"
        "Formulate the answer naturally in your own words.\n"
        "Do not repeat stored sentences verbatim.\n"
        "Provide only the direct answer.\n\n"
        "Known information:\n"
        + context_block +
        "\n\nQuestion:\n"
        + question.strip() +
        "\n\nAnswer:\n"
    )


# =========================================================
# ACADEMIC PROMPT
# =========================================================
# Used for retrieval-backed academic explanations.
# Retrieval injection logic:
#   - Accepts ordered `chunks` from upstream retrieval.
#   - Each chunk is stripped and inserted as numbered source context: `[n] ...`.
# Prompt component order:
#   1) `SYSTEM_IDENTITY`
#   2) Academic response policy and structure constraints
#   3) Injected source-material block
#   4) User question
#   5) Assistant cue ("Answer:")

def build_academic_prompt(question: str, chunks: List[str]) -> str:
    """Build an academic prompt with explicit source-only constraints.

    Args:
        question: Final user question text.
        chunks: Retrieved source chunks to be used as context evidence.

    Returns:
        Fully assembled academic prompt string.

    Determinism:
        Deterministic for identical `question` and ordered `chunks`.

    Edge cases:
        - Empty `chunks` produces an empty source section (header remains).
        - Each chunk is trimmed before insertion.

    Failure handling:
        No local exception handling is applied. Invalid chunk entries that do not
        support `.strip()` raise upstream-visible exceptions.

    Prompt safety considerations:
        Source chunks are inserted verbatim as untrusted text; the prompt relies
        on policy instructions ("use only provided source material") rather than
        structural sanitization.
    """

    base_instruction = (
        "You are a technical instructor.\n"
        "Answer exclusively in the language of the question.\n"
        "Use only the provided source material as your knowledge base.\n"
        "Explain concepts in your own words.\n"
        "Do not quote raw source text.\n"
        "Structure your response as:\n"
        "1. Short definition\n"
        "2. Technical explanation\n"
        "3. Practical example\n"
        "If information is missing, respond exactly with:\n"
        "Insufficient information in memory.\n\n"
    )

    context_block = "\n\n".join(
        f"[{i+1}] {c.strip()}"
        for i, c in enumerate(chunks)
    )

    return (
        SYSTEM_IDENTITY +
        base_instruction +
        "Source material (do not quote directly):\n"
        + context_block +
        "\n\nQuestion:\n"
        + question.strip() +
        "\n\nAnswer:\n"
    )


# =========================================================
# GENERAL PROMPT
# =========================================================
# Used as fallback/general route prompt.
# Prompt component order:
#   1) `SYSTEM_IDENTITY`
#   2) General tutor instruction block
#   3) User question
#   4) Assistant cue ("Explanation:")

def build_general_prompt(question: str) -> str:
    """Build a general-knowledge prompt without memory/retrieval injection.

    Args:
        question: Final user question text.

    Returns:
        Fully assembled general prompt string.

    Determinism:
        Deterministic for identical `question`.

    Edge cases:
        `question` is stripped before insertion.

    Failure handling:
        No local exception handling; unexpected `question` types are surfaced
        through standard Python string-conversion/attribute errors.

    Prompt safety considerations:
        This prompt injects only user question text and static instructions.
        It does not include retrieved memory/source content.
    """

    return (
        SYSTEM_IDENTITY +
        "You are a technical tutor.\n"
        "Answer using your general knowledge.\n"
        "Provide a clear and structured explanation.\n\n"
        "Question:\n"
        + question.strip() +
        "\n\nExplanation:\n"
    )
