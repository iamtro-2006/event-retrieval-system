# src/retrieval/indexer/__init__.py
from .faiss.index_builder import (
    build_matrix_and_metadata,
    create_faiss_index,
    save_faiss_index,
)

__all__ = [
    "build_matrix_and_metadata",
    "create_faiss_index",
    "save_faiss_index",
]
