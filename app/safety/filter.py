"""Rule-based lexical safety gate.

Purpose:
    Provide a deterministic pre-generation check that can block clear misuse
    requests while still allowing educational/theoretical questions.

Validation model:
    - Rule-based only (substring matching), no classifier/model inference.
    - Allow-list patterns are evaluated before block-list patterns.
    - Output is a boolean gate consumed by orchestration (`app.core`).

Blocking behavior:
    - `True`: request is allowed to continue to routing/prompting.
    - `False`: caller should short-circuit and return a policy response.

Determinism:
    For the same input text and pattern lists, output is deterministic.

Failure handling:
    This module does not raise policy-specific exceptions; invalid/empty input
    handling is minimal and explicit in `is_allowed`.

Bypass risk:
    Substring matching can be bypassed by obfuscation, misspellings, spacing
    tricks, or unsupported languages. Upstream and downstream controls are still
    required for defense in depth.

Performance:
    Runtime is linear in the number of configured patterns and input length, with
    no external I/O.
"""

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
    """Return whether a question should pass the lexical safety gate.

    Args:
        question: Raw user text to validate.

    Returns:
        `True` when input is considered safe enough to continue processing;
        `False` when blocked-pattern evidence is detected.

    Evaluation order:
        1. Empty input is allowed.
        2. Any theoretical/educational allow pattern -> allow immediately.
        3. Any blocked misuse pattern -> block.
        4. Otherwise allow.

    Rule-based vs model-based:
        Entirely rule-based lexical matching; no model scoring or probabilistic
        thresholds are used.

    Interaction with core/prompting:
        Callers in orchestration run this before prompt assembly/model calls.
        A `False` result is expected to produce a direct policy response.

    Determinism:
        Deterministic for identical input and constant pattern lists.

    Failure handling:
        Empty/None-like values are treated as allowed; no exceptions are handled
        here beyond Python runtime behavior.

    Bypass considerations:
        Lexical checks are intentionally simple and may miss paraphrased or
        obfuscated malicious intent.

    Performance:
        Two `any(...)` scans over short static pattern lists (constant-space).
    """
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
