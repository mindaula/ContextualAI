from typing import List


# =========================================================
# SYSTEM IDENTITY (GLOBAL)
# =========================================================
# Defines the global behavioral constraints of the assistant.
# Keeps the model neutral, local, and consistent.

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
# Used when answering questions about stored personal facts.
# Prevents verbatim repetition of stored memory entries.

def build_personal_prompt(question: str, facts) -> str:

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
# Used for structured technical or academic explanations.
# Enforces structured teaching format.

def build_academic_prompt(question: str, chunks: List[str]) -> str:

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
# Used for general knowledge or fallback cases.

def build_general_prompt(question: str) -> str:

    return (
        SYSTEM_IDENTITY +
        "You are a technical tutor.\n"
        "Answer using your general knowledge.\n"
        "Provide a clear and structured explanation.\n\n"
        "Question:\n"
        + question.strip() +
        "\n\nExplanation:\n"
    )
