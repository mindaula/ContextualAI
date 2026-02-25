"""Embedding model bootstrap for the memory subsystem.

Architectural role:
    Provides a single shared `SentenceTransformer` instance used by memory and NLP
    components for vectorization. The loader decides CPU vs CUDA execution once and
    reuses the initialized model across subsequent calls.

Design intent:
    - Keep embedding initialization centralized.
    - Avoid duplicated model loads across modules.
    - Apply a conservative VRAM gate before enabling GPU execution.
"""

import os

EMBED_MODEL = "intfloat/multilingual-e5-small"
_model = None


def has_enough_vram(min_required_mb: int = 800) -> bool:
    """Return whether enough free GPU memory is available for embeddings.

    Args:
        min_required_mb: Minimum required free VRAM in megabytes.

    Returns:
        `True` when CUDA is available and free VRAM exceeds the threshold.

    Side effects:
        Prints detected free VRAM when CUDA is available.
    """
    import torch

    if not torch.cuda.is_available():
        return False

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_mb = free_mem / 1024 / 1024

    print(f"Free VRAM: {free_mb:.0f} MB")

    return free_mb > min_required_mb


def get_model():
    """Load and cache the shared embedding model instance.

    Returns:
        A `SentenceTransformer` instance configured for CUDA or CPU.

    Behavior:
        - Uses singleton caching via module-global `_model`.
        - Enables CUDA only when `has_enough_vram()` returns `True`.
        - Forces CPU mode by setting `CUDA_VISIBLE_DEVICES=""` otherwise.

    Side effects:
        - Imports `torch`/`sentence_transformers` lazily.
        - Updates `os.environ["CUDA_VISIBLE_DEVICES"]` in CPU fallback mode.
        - Emits startup diagnostics via `print`.
    """
    global _model

    if _model is not None:
        return _model

    print("Loading embedding model...")

    use_gpu = False

    try:
        import torch
        use_gpu = has_enough_vram()
    except Exception:
        use_gpu = False

    if not use_gpu:
        print("Insufficient VRAM detected. Forcing CPU mode.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from sentence_transformers import SentenceTransformer

    device = "cuda" if use_gpu else "cpu"

    print(f"Loading embeddings on {device.upper()}")

    _model = SentenceTransformer(EMBED_MODEL, device=device)

    try:
        print("Embedding device:", next(_model.parameters()).device)
    except Exception:
        pass

    return _model
