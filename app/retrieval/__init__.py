"""Retrieval package.

Architectural role:
    Provides retrieval-time context assembly, memory-search adapters, ingestion
    utilities for academic memory, and web context acquisition used by core routing.

Scope:
    - `context_builder`: route-scoped retrieval context composition.
    - `retriever`: personal/academic retrieval adapters.
    - `ingestion`: offline data ingestion into academic memory.
    - `web`: external web search/extraction pipeline for untrusted context.
"""
