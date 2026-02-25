"""ContextualAI API adapter package.

Architectural role:
- Defines the external interaction boundary for HTTP and CLI interfaces.
- Performs transport-level validation and response shaping.
- Delegates reasoning/orchestration to the core layer.

Scope:
- Request lifecycle control for adapter concerns only.
- No direct model invocation logic is implemented in this package root.
"""
