"""Rule-based structural classifier for user utterances.

Parsing rules:
- Scores greeting/meta/question/statement signals using multilingual token sets.
- Uses punctuation and first-token interrogative heuristics for question detection.
- Returns the best-scoring structural label or `unknown`.

Normalization steps:
- Lowercasing and accent folding for DE/ES/FR variants.
- Regex tokenization with word boundaries.

Determinism:
- Fully deterministic given identical input and static vocabulary sets.

Temporal handling:
- None. Classification uses only the current utterance.

Edge cases:
- Empty/whitespace-only input yields `unknown`.
- Mixed signals are resolved by score maximum; no tie-break randomness.
"""

import re


# =========================================================
# STRUCTURAL CLASSIFIER (Stable Hybrid Version)
# =========================================================
# Detects STRUCTURE only (not semantic intent)
#
# Output categories:
# - greeting
# - statement
# - question
# - meta
# - unknown
# =========================================================


# ---------------------------------------------------------
# Greeting vocabulary (multilingual, short-only)
# ---------------------------------------------------------

GREETING_WORDS = {
    # German
    "hallo", "moin", "servus",

    # English
    "hi", "hello", "hey",

    # Spanish
    "hola",

    # French
    "bonjour", "salut"
}


# ---------------------------------------------------------
# Meta keywords
# ---------------------------------------------------------

META_KEYWORDS = {
    "router", "model", "modell", "system",
    "chat", "conversation", "engine",
    "memory", "intent"
}


# ---------------------------------------------------------
# Interrogative tokens (DE / EN / ES / FR)
# ---------------------------------------------------------

QUESTION_TOKENS = {

    # -------------------------
    # German
    # -------------------------
    "wer", "was", "wann", "wo", "warum",
    "wieso", "weshalb", "wie", "welche",
    "welcher", "welches",

    # German compound interrogatives
    "worueber", "worüber",
    "woran",
    "womit",
    "wodurch",
    "wofuer", "wofür",

    # -------------------------
    # English
    # -------------------------
    "who", "what", "when", "where",
    "why", "how", "which",

    # -------------------------
    # Spanish
    # -------------------------
    "quien", "quién",
    "que", "qué",
    "cuando", "cuándo",
    "donde", "dónde",
    "por",          # por qué
    "como", "cómo",
    "cual", "cuál",
    "cuales", "cuáles",

    # -------------------------
    # French
    # -------------------------
    "qui", "quoi",
    "quand",
    "ou", "où",
    "pourquoi",
    "comment",
    "quel", "quelle",
    "quels", "quelles"
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize(text: str) -> str:
    """Apply lightweight multilingual normalization before token scoring."""
    text = text.strip().lower()

    # German umlauts
    text = (
        text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
    )

    # French/Spanish accents (minimal safe normalization)
    text = (
        text.replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("á", "a")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
    )

    return text


def _tokenize(text: str):
    """Tokenize normalized text into word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# =========================================================
# MAIN CLASSIFIER
# =========================================================

def classify_structure(text: str) -> str:
    """
    Classify utterance structure (`greeting|statement|question|meta|unknown`).

    Failure handling:
    - Empty/invalid tokenized input returns `unknown`.
    """

    if not text or not text.strip():
        return "unknown"

    original = text.strip()
    q = _normalize(original)
    tokens = _tokenize(q)

    if not tokens:
        return "unknown"

    token_count = len(tokens)

    # -----------------------------------------------------
    # Feature scoring
    # -----------------------------------------------------

    scores = {
        "greeting": 0.0,
        "meta": 0.0,
        "question": 0.0,
        "statement": 0.0
    }

    # -----------------------------------------------------
    # Greeting detection (short only)
    # -----------------------------------------------------

    if token_count <= 3 and all(t in GREETING_WORDS for t in tokens):
        scores["greeting"] += 1.0

    # -----------------------------------------------------
    # Meta detection
    # -----------------------------------------------------

    meta_hits = sum(1 for t in tokens if t in META_KEYWORDS)
    if meta_hits > 0:
        scores["meta"] += 0.7 + 0.1 * meta_hits

    # -----------------------------------------------------
    # Question detection
    # -----------------------------------------------------

    # Explicit question mark
    if original.endswith("?"):
        scores["question"] += 1.0

    # Interrogative first token
    if tokens[0] in QUESTION_TOKENS:
        scores["question"] += 0.8

    # Inverted verb heuristic (DE/EN core)
    if token_count >= 2:
        first_two = " ".join(tokens[:2])
        if re.match(r"^(ist|sind|hast|habt|kann|kannst|can|do|does|did)\b", first_two):
            scores["question"] += 0.6

    # -----------------------------------------------------
    # Statement heuristic
    # -----------------------------------------------------

    if token_count >= 2:
        scores["statement"] += 0.5

    if token_count == 1:
        scores["statement"] += 0.2

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score <= 0:
        return "unknown"

    return best_label
