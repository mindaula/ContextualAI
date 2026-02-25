"""Core orchestration package.

Architectural role:
    Exposes the request-orchestration layer that sits between API/CLI entrypoints
    and lower-level subsystems (routing, retrieval, prompting, memory, and LLM
    adapters).

Composition:
    - `engine`: Main control-flow implementation for request processing.
    - `routing_types`: Shared routing decision schema consumed by NLP router logic.

Determinism and side effects:
    Package import itself is deterministic and side-effect free. Runtime side effects
    are performed by `engine` during request processing.
"""
