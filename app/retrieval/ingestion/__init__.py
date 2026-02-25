"""Academic ingestion subpackage.

Architectural role:
    Hosts CLI ingestion tooling that converts local documents/repositories into
    normalized chunks and metadata for storage in academic memory indexes.

FAISS interaction:
    No direct index operations in package init; write-path logic is delegated to
    `app.memory.memory_system` by ingestion modules.
"""
