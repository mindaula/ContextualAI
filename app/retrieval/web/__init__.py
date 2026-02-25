"""Web retrieval subpackage.

Architectural role:
    Provides provider-agnostic web search and extraction components used by core
    manual web-search flows.

Security model:
    Extracted web content is treated as untrusted context and sanitized before it is
    returned to prompt-construction layers.
"""
