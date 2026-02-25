"""Priority-based safety/context decision helper.

Current status:
    This module defines a small rule-based decision object/selector and is
    currently optional in the active orchestration path.

Decision model:
    - Rule-based only (no model inference).
    - Chooses the highest-priority available context bucket.

Priority order:
    1. personal hits
    2. academic hits
    3. general fallback

Determinism:
    Deterministic for identical inputs.

Failure handling:
    No internal exception handling; assumes caller passes list-like collections.

Performance:
    Constant-time checks over already-computed inputs.
"""

class Decision:
    """Container for selected decision origin and attached context payload."""

    def __init__(self, origin, context=None):
        self.origin = origin
        self.context = context or []


def decide(route, personal_hits, academic_hits):
    """Select a context origin using fixed priority rules.

    Args:
        route: Route hint provided by caller (currently unused by logic).
        personal_hits: Retrieved personal-memory hits.
        academic_hits: Retrieved academic-memory hits.

    Returns:
        `Decision` with origin in {"personal", "academic", "general"} and
        optional context payload.

    Blocking vs filtering behavior:
        This function does not block requests; it only selects which context
        source should drive downstream response behavior.

    Rule-based vs model-based:
        Fully rule-based deterministic branching with no model dependency.

    Interaction with core/prompting:
        If used, caller can map `origin` to prompt strategy/context injection.

    Bypass risk:
        Because selection is presence-based only, incorrect upstream retrieval
        can bias chosen origin.
    """
    # Personal context has the highest priority when available.
    if personal_hits:
        return Decision("personal", personal_hits)

    # Academic context is selected when personal context is absent.
    if academic_hits:
        return Decision("academic", academic_hits)

    # Otherwise fall back to generic behavior without retrieval context.
    return Decision("general")
