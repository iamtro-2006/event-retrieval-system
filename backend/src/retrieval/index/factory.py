from __future__ import annotations

from src.retrieval.index.clip_faiss_index import ClipFaissIndex


def build_clip_faiss_index(config: dict) -> ClipFaissIndex:
    """Build a `ClipFaissIndex` từ config dict (index_path, metadata_path,
    model_name, pretrained, device, precision, ... — xem `ClipFaissIndex.__init__`).
    """
    return ClipFaissIndex(**config)
