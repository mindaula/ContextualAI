import re
from app.llm_interface import generate_answer


# =========================================================
# MULTILINGUAL PRONOUN / FOLLOW-UP DETECTION
# =========================================================
# Detects context-dependent references across multiple languages:
# German, English, Spanish, French

PRONOUN_PATTERN = re.compile(
    r"\b("
    # German
    r"davon|dafuer|dafür|dazu|darueber|darüber|damit|"
    r"das|es|dies|diese|dieser|dieses|jenes|jene|"
    r"denen|ihnen|daran|darauf|dessen|"
    # English
    r"it|this|that|these|those|them|they|"
    r"thereof|therefor|therefore|about\sit|of\sit|"
    # Spanish
    r"eso|esa|ese|esas|esos|ello|ellos|ellas|"
    r"de\s+eso|sobre\s+eso|"
    # French
    r"cela|ça|ca|ceci|celui|celle|ceux|celles|"
    r"en|y|à\s+cela|de\s+cela"
    r")\b",
    flags=re.IGNORECASE
)


def _has_pronoun(text: str) -> bool:
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
    return word_count <= 3


def _clean_llm_output(text: str) -> str:
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

    if not question or not question.strip():
        return question

    question = question.strip()

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
            last_route == "academic"
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
