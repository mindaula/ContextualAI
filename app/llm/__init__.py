"""LLM access package.

Architectural role:
    Provides provider configuration, request-payload construction, and transport
    adapters used by orchestration layers to invoke text-generation backends.

Module split:
    - `provider_config`: environment-driven provider and model configuration.
    - `service`: canonical prompt-to-payload adapter.
    - `client`: provider-specific HTTP transport and response parsing.
"""
