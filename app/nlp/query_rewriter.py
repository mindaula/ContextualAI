"""Follow-up query rewriting for context-dependent user inputs.

Intent logic:
- Detects referential follow-ups via multilingual pronoun matching and short-form
  heuristics.
- Uses temporal context (`last_route` + recent user messages) to decide whether
  rewrite is required.

Interaction with core and memory:
- Called from core before route execution.
- Reads short-term conversation history via `conversation_manager.get_recent_messages`.
- Uses semantic-role classifier to protect identity/memory queries from rewrite.

Determinism:
- Rewrite trigger checks and fallback formatting are deterministic.
- LLM rewrite output is model-dependent and therefore non-deterministic.

Failure handling:
- History-fetch and LLM failures degrade to no-rewrite or deterministic fallback.
"""

import re
from app.llm.service import generate_answer
from app.nlp.semantic_role_classifier import classify_semantic_role  # ← MINIMAL FIX


# =========================================================
# MULTILINGUAL PRONOUN / FOLLOW-UP DETECTION
# =========================================================
# Detects context-dependent references across multiple languages:
# German, English, Spanish, French

PRONOUN_PATTERN = re.compile(
    r"\b("
    # German
    r"davon|dafuer|dafür|dazu|darueber|darüber|damit|"
    r"das|dies|diese|dieser|dieses|jenes|jene|"
    r"denen|ihnen|daran|darauf|dessen|"
    # English
    r"it|this|that|these|those|them|they|"
    r"thereof|therefor|therefore|about\sit|of\sit|"
    # Spanish
    r"eso|esa|ese|esas|esos|ello|ellos|ellas|"
    r"de\s+eso|sobre\s+eso|"
    # French
    r"cela|ça|ca|ceci|celui|celle|ceux|celles|"
    r"à\s+cela|de\s+cela"
    r")\b",
    flags=re.IGNORECASE
)


def _has_pronoun(text: str) -> bool:
    """Return `True` when text contains a known referential pronoun pattern."""
    if not text:
        return False
    return bool(PRONOUN_PATTERN.search(text))


def _is_short_followup(text: str) -> bool:
    """
    Only extremely short follow-up questions are considered context-dependent.
    Example:
    - "and why?"
    - "which one?"
    - "is there more?"
    """
    if not text:
        return False

    word_count = len(text.strip().split())
    return word_count <= 8


def detects_referential_followup(text: str) -> bool:
    """Detect referential follow-ups using pronoun + short-length conjunction."""
    return _has_pronoun(text) and _is_short_followup(text)


def _clean_llm_output(text: str) -> str:
    """Normalize rewrite output to a single trimmed line."""
    if not text:
        return ""

    line = text.strip().splitlines()[0].strip()
    line = line.strip('"').strip("'").strip()
    return line


# =========================================================
# MAIN REWRITE FUNCTION
# =========================================================
# Rewrites context-dependent user queries into complete standalone questions.

def rewrite_query(
    question: str,
    conversation_manager,
    last_route: str = None,
    max_lookback: int = 6,
    force_rewrite: bool = False
) -> str:
    """
    Rewrite context-dependent questions into standalone form when needed.

    Parsing and decision rules:
    - Skips rewrite for blank inputs.
    - Skips rewrite for semantic roles `self_query` and `self_disclosure`.
    - Triggers rewrite when:
      - referential pronouns are present, or
      - short follow-up appears after academic/manual web-search route.

    Temporal handling:
    - Uses `last_route` and recent user messages (`max_lookback`) to provide
      rewrite context and trigger conditions.

    Edge cases:
    - If rewrite is needed but no prior user question is available, original
      question is returned unchanged.
    - If LLM output is too short/invalid, deterministic fallback string is used.
    """

    if not question or not question.strip():
        return question

    question = question.strip()

    # -----------------------------------------------------
    # 🔒 SEMANTIC GUARD (MINIMAL FIX)
    # Prevent rewriting of identity / memory related queries
    # -----------------------------------------------------

    role, role_score = classify_semantic_role(question)

    if role in ["self_query", "self_disclosure"]:
        return question

    # -----------------------------------------------------
    # Retrieve last user question
    # -----------------------------------------------------

    try:
        recent = conversation_manager.get_recent_messages(limit=max_lookback) or []
    except Exception:
        recent = []

    last_user = None

    for msg in reversed(recent):
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "user":
            content = msg.get("content", "").strip()

            if content and content != question:
                last_user = content
                break

    # -----------------------------------------------------
    # Determine whether rewrite is required
    # -----------------------------------------------------

    needs_rewrite = force_rewrite

    if not needs_rewrite:

        # Case 1: Pronoun detected → rewrite
        if _has_pronoun(question):
            needs_rewrite = True

        # Case 2: Very short follow-up in academic context
        elif (
            last_route in ["academic", "manual_web_search"]
            and last_user
            and _is_short_followup(question)
        ):
            needs_rewrite = True

    # No previous user message available → skip rewrite
    if needs_rewrite and not last_user:
        return question

    if not needs_rewrite:
        return question

    # -----------------------------------------------------
    # LLM Rewrite
    # -----------------------------------------------------

    prompt = f"""
You are a precise assistant that reformulates user questions.

Goal:
Rewrite the current question so that it is fully self-contained,
explicit, and does not contain ambiguous pronouns such as
"it", "that", "this", "those", etc.

If helpful, incorporate context from the previous user question.

Previous user question:
{last_user}

Current question:
{question}

Return ONLY the fully reformulated question.
One single line. No explanation.
"""

    try:
        rewritten = generate_answer(prompt, stream=False)
    except Exception:
        rewritten = None

    rewritten = _clean_llm_output(rewritten)

    # Safety check against hallucinated unrelated rewrites
    if rewritten and len(rewritten.split()) >= len(question.split()):
        return rewritten

    # -----------------------------------------------------
    # Fallback (without LLM)
    # -----------------------------------------------------

    return f"{question} (in reference to: {last_user})"
