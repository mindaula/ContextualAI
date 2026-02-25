"""Memory subsystem package.

Architectural role:
    Groups the stateful memory components used by the application:
    - `embedding_model`: shared embedding model bootstrap/singleton.
    - `memory_system`: personal and academic knowledge memory storage/retrieval.
    - `conversation_memory`: long-term conversational memory indexing/retrieval.
    - `conversation_manager`: short-term session buffering, crash recovery, archive.

This package intentionally centralizes persistence-facing memory behavior so routing,
API, and retrieval layers can depend on a stable memory interface.
"""
