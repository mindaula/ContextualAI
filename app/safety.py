# =========================================================
# SAFETY FILTER
# =========================================================
# Basic keyword-based safety filter.
# Blocks direct misuse attempts while allowing theoretical discussion.

BLOCKED_PATTERNS = [

    # English
    "how to hack",
    "illegal access",
    "bypass authentication",
    "exploit live target",
    "break into account",

    # German
    "wie hacke ich",
    "illegaler zugriff",
    "authentifizierung umgehen",
    "in account einbrechen",

    # Spanish
    "como hackear",
    "acceso ilegal",
    "omitir autenticacion",
    "entrar en cuenta ajena",

    # French
    "comment pirater",
    "acces illegal",
    "contourner authentification",
    "pirater un compte"
]


ALLOWED_THEORETICAL_PATTERNS = [

    # English
    "how does",
    "how do attacks work",
    "explain",

    # German
    "wie funktioniert",
    "erklaere",
    "erkläre",

    # Spanish
    "como funciona",
    "explica",

    # French
    "comment fonctionne",
    "explique"
]


def is_allowed(question: str) -> bool:
    if not question:
        return True

    q = question.lower()

    # Allow theoretical explanations
    if any(p in q for p in ALLOWED_THEORETICAL_PATTERNS):
        return True

    # Block direct misuse attempts
    if any(p in q for p in BLOCKED_PATTERNS):
        return False

    return True
