"""NLP utilities for routing, query rewriting, and lightweight classification.

Module scope:
- Intent/routing signal extraction (`intent_router`).
- Structure and semantic-role parsing (`structure_classifier`, `semantic_role_classifier`).
- Follow-up query rewriting (`query_rewriter`).
- Auxiliary fact/knowledge-need classification (`fact_classifier`).

Determinism profile:
- Mix of deterministic rule logic and model-backed scoring/generation.
"""
