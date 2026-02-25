"""Routing decision data contracts for `app.core.engine`.

Architectural role:
    Defines the minimal schema returned by the intent router and consumed by the
    orchestration engine when selecting request routes.

Control-flow interaction:
    `engine.process_message` inspects these fields in fixed priority order to map a
    classification result to a concrete route (`personal_store`, `personal_query`,
    `academic`, `conversation_query`, `followup_transform`, or fallback route).

Determinism:
    The data class is purely structural and state-free. Determinism depends on the
    classifier modules that populate it, not on this module.
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Normalized route selection flags produced by the NLP router.

    Attributes:
        fallback: Default route label when no specialized flag is active.
        confidence: Router confidence score propagated into memory-context handling.
        store_personal_fact: Enables personal-memory write flow.
        use_long_term_memory: Enables personal-memory retrieval flow.
        use_academic_chunks: Enables academic retrieval flow.
        use_short_term_memory: Enables conversation-recall retrieval flow.
        followup_transform: Enables follow-up transformation of the last answer.
    """

    fallback: str | None = None
    confidence: float = 0.0

    store_personal_fact: bool = False
    use_long_term_memory: bool = False
    use_academic_chunks: bool = False
    use_short_term_memory: bool = False

    followup_transform: bool = False
