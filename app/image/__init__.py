"""Image generation adapter package.

Scope:
    Provides text-to-image provider clients and a small dispatch service used by
    core orchestration when `/image ...` commands are issued.

Non-goals:
    - No file ingestion or multimodal file analysis.
    - No Base64 decoding/encoding pipeline.
    - No temporary-file creation or cleanup responsibilities.
"""
