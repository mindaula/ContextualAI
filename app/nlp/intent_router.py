"""Intent router producing `RoutingDecision` for core orchestration.

Intent classification logic:
- Parses explicit hard prefixes (`/general`, `/academic`) first.
- Applies temporal follow-up transform detection via `last_route` + marker list.
- Uses structural parsing (`classify_structure`) and semantic-role classification
  (`classify_semantic_role`) to map input into decision flags.

Interaction with core and memory:
- Returns `RoutingDecision` consumed by `app.core.engine`.
- Sets flags that core maps to memory read/write flows:
  - `store_personal_fact` -> personal-store path.
  - `use_long_term_memory` -> personal query path.
  - `use_short_term_memory` -> conversation-memory path.

Temporal handling:
- `last_route` is used to detect transform-style follow-ups tied to prior route.

Determinism:
- Deterministic for fixed classifier outputs and identical input.
- Overall behavior depends on `classify_semantic_role`, which is model-backed.

Failure handling:
- Empty/blank input falls back to `general`.
"""

from app.core.routing_types import RoutingDecision
from app.nlp.structure_classifier import classify_structure
from app.nlp.semantic_role_classifier import classify_semantic_role


# =========================================================
# FOLLOWUP MARKERS (MINIMAL ADDITION)
# =========================================================

FOLLOWUP_MARKERS = [
    "nochmal",
    "wiederholen",
    "repeat",
    "explain again",
    "auf deutsch",
    "auf englisch",
    "kürzer",
    "ausführlicher"
]


def decide_route(question: str, last_route: str | None = None) -> RoutingDecision:
    """
    Classify a user question into routing flags and fallback route.

    Parsing rules:
    1. Hard trigger prefixes are handled first (`/general`, `/academic`).
    2. Follow-up transform markers are checked using temporal context.
    3. Structural and semantic classifications are combined into final flags.

    Edge cases:
    - Blank question -> `fallback = "general"`.
    - Non-question self-disclosure is forced into personal-store flow.
    - Generic questions default to `academic` when no earlier rule matches.
    """

    decision = RoutingDecision()

    if not question or not question.strip():
        decision.fallback = "general"
        return decision

    raw_question = question
    stripped = question.strip()
    prefix = stripped.lower().split(" ", 1)[0]

    if prefix == "/general":
        cleaned_question = stripped[len(prefix):].strip()
        decision.fallback = "general"
        decision.cleaned_question = cleaned_question
        return decision

    if prefix == "/academic":
        cleaned_question = stripped[len(prefix):].strip()
        decision.fallback = "academic"
        decision.cleaned_question = cleaned_question
        return decision

    # =====================================================
    # FOLLOWUP TRANSFORM (MINIMAL EARLY EXIT)
    # =====================================================

    if (
        last_route in ["academic", "general", "manual_web_search"]
        and any(marker in question.lower() for marker in FOLLOWUP_MARKERS)
    ):
        decision.followup_transform = True
        return decision

    # -----------------------------------------------------
    # STAGE 1: STRUCTURE
    # -----------------------------------------------------

    structure = classify_structure(question)

    # -----------------------------------------------------
    # STAGE 2: SEMANTIC ROLE
    # -----------------------------------------------------

    role, role_score = classify_semantic_role(question)
    decision.confidence = role_score

    def _debug_decision_state() -> None:
        print(
            "[ROUTER DEBUG] "
            f"role={role}, "
            f"structure={structure}, "
            f"decision.store_personal_fact={decision.store_personal_fact}"
        )

    # -----------------------------------------------------
    # SELF DISCLOSURE → STORE
    # -----------------------------------------------------

    if role == "self_disclosure" and structure != "question":
        decision.store_personal_fact = True
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # SELF QUERY → LONG TERM MEMORY
    # -----------------------------------------------------

    if role == "self_query" and structure == "question":
        decision.use_long_term_memory = True
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # CONVERSATION REFERENCE → SHORT TERM MEMORY
    # -----------------------------------------------------

    if role == "conversation_reference":
        decision.use_short_term_memory = True
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # KNOWLEDGE QUESTION → ACADEMIC (MINIMAL FIX)
    # -----------------------------------------------------

    if role == "knowledge_question":
        decision.fallback = "academic"
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # SMALLTALK → GENERAL
    # -----------------------------------------------------

    if role == "smalltalk":
        decision.fallback = "general"
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # GENERIC QUESTIONS → ACADEMIC (MINIMAL FIX)
    # -----------------------------------------------------

    if structure == "question":
        decision.fallback = "academic"
        _debug_decision_state()
        return decision

    # -----------------------------------------------------
    # DEFAULT
    # -----------------------------------------------------

    decision.fallback = "general"
    _debug_decision_state()
    return decision
