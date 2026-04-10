"""
palace.py — Shared palace operations.

Consolidates ChromaDB access patterns used by both miners and the MCP server.
"""

import logging
import os

import chromadb

from .config import (
    EmbeddingModelMismatchError,
    MempalaceConfig,
    get_embedding_function,
    get_embedding_model_name,
)

logger = logging.getLogger("mempalace")

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}


def get_collection(palace_path: str, collection_name: str = "mempalace_drawers", force: bool = None):
    """Get or create the palace ChromaDB collection.

    Verifies that the collection's embedding model matches the currently
    configured model. Raises EmbeddingModelMismatchError on mismatch
    unless force=True or MEMPALACE_FORCE_EMBEDDING=true.
    """
    os.makedirs(palace_path, exist_ok=True)
    try:
        os.chmod(palace_path, 0o700)
    except (OSError, NotImplementedError):
        pass

    ef = get_embedding_function()
    current_model = get_embedding_model_name()
    if force is None:
        force = MempalaceConfig().force_embedding

    client = chromadb.PersistentClient(path=palace_path)
    try:
        col = client.get_collection(collection_name, embedding_function=ef)
        stored_model = (col.metadata or {}).get("embedding_model")

        if stored_model is None:
            # Legacy palace — silent stamp
            col.modify(metadata={**(col.metadata or {}), "embedding_model": current_model})
        elif stored_model != current_model:
            if force:
                logger.warning(
                    "Embedding model mismatch (forced): %s -> %s",
                    stored_model, current_model,
                )
                col.modify(metadata={**(col.metadata or {}), "embedding_model": current_model})
            else:
                raise EmbeddingModelMismatchError(stored_model, current_model)

        return col
    except EmbeddingModelMismatchError:
        raise
    except Exception:
        return client.create_collection(
            collection_name,
            embedding_function=ef,
            metadata={"embedding_model": current_model},
        )


def iter_all_metadatas(collection, where=None, page_size: int = 10000):
    """Yield every metadata entry in the collection, paginating past the page cap.

    ChromaDB's ``collection.get()`` enforces a per-call limit, so a single fetch
    silently truncates large palaces. This walks the collection in pages so
    callers see every drawer — with or without a ``where`` filter.
    """
    offset = 0
    while True:
        kwargs = {"include": ["metadatas"], "limit": page_size, "offset": offset}
        if where is not None:
            kwargs["where"] = where
        page = collection.get(**kwargs)
        metas = page.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            yield m
        offset += len(metas)


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    When check_mtime=True (used by project miner), returns False if the file
    has been modified since it was last mined, so it gets re-mined.
    When check_mtime=False (used by convo miner), just checks existence.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        if check_mtime:
            stored_meta = results.get("metadatas", [{}])[0]
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return float(stored_mtime) == current_mtime
        return True
    except Exception:
        return False
