"""IndexManager — quản lý NHIỀU `FaissIndex` cùng lúc (1 model = 1 FaissIndex).

Đây là chỗ trả lời câu hỏi #2 (README người dùng): "giúp tôi có thể chọn mô
hình nào để load cho semantic/temporal". Thay vì `system.py` build đúng 1
`FaissIndex` (1 model) như bản cũ, giờ nó build 1 `IndexManager` từ 1 LIST
model trong config (`configs/indexing.yaml` -> `models: [...]`, cùng hình
dạng với `configs/embeddings.yaml`), mỗi entry là 1 model + FAISS index
riêng của nó. `semantic_search`/`temporal_search` pipeline nhận
`IndexManager` này và cho phép chọn `model_key` mỗi lần gọi `.search(...)`
(mặc định dùng `default_model_key` nếu không truyền).

Model không load được (thiếu checkpoint, thiếu index...) sẽ bị SKIP với 1
cảnh báo in ra console, thay vì làm sập toàn bộ hệ thống — giống cách
`system.py` đã làm với OCR/ASR optional subsystem.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.retrieval.index.faiss_index import FaissIndex


class IndexManager:
    """Dict-like registry of `{model_key: FaissIndex}`."""

    def __init__(
        self, indexes: dict[str, FaissIndex], default_model_key: str | None = None
    ) -> None:
        if not indexes:
            raise ValueError(
                "IndexManager needs at least 1 successfully loaded FaissIndex."
            )
        self._indexes = indexes
        self._default_model_key = default_model_key or next(iter(indexes))
        if self._default_model_key not in self._indexes:
            raise ValueError(
                f"default_model_key '{self._default_model_key}' not among loaded models: "
                f"{sorted(self._indexes)}"
            )

    @property
    def default_model_key(self) -> str:
        return self._default_model_key

    def keys(self) -> list[str]:
        """All successfully-loaded model keys, e.g. for populating the
        `advanced_search` checklist UI."""
        return list(self._indexes.keys())

    def text_search_keys(self) -> list[str]:
        """Subset of `keys()` whose model actually supports text queries
        (some backends, e.g. BEiT-3 for now, are image-only — see
        `embedding_extraction/models/backends/beit3.py`)."""
        return [key for key, idx in self._indexes.items() if idx.supports_text]

    def get(self, model_key: str | None = None) -> FaissIndex:
        key = model_key or self._default_model_key
        try:
            return self._indexes[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model_key '{key}'. Loaded models: {sorted(self._indexes)}"
            ) from exc

    def __contains__(self, model_key: str) -> bool:
        return model_key in self._indexes

    def __iter__(self) -> Iterable[str]:
        return iter(self._indexes)

    def items(self):
        return self._indexes.items()


def build_index_manager(
    config: dict[str, Any] | list[dict[str, Any]],
    milvus_cfg: dict[str, Any] | None = None,
) -> IndexManager:
    """Build an `IndexManager` from config.

    Args:
        config: Either
            - a list of per-model dicts (each a kwargs dict for
              `FaissIndex.__init__` or `ClipMilvusIndex.__init__`, must include
              a unique `model_key`), or
            - a dict with keys `{"models": [...], "default_model_key": "..."}`
              (matches `configs/indexing.yaml` shape: `semantic.models` /
              `temporal.models`).
        milvus_cfg: Optional top-level `milvus:` connection block
            (`{"host": ..., "port": ...}`). Injected as `milvus_host` /
            `milvus_port` into any model entry whose `backend` is `"milvus"`.

    A model entry with `backend: "milvus"` builds a `ClipMilvusIndex` (Milvus
    ANN) instead of a `FaissIndex`; every other entry builds a `FaissIndex`.

    Models whose entry has `enabled: false`, or that raise on load (missing
    checkpoint/index file/etc.), are skipped with a printed warning rather
    than aborting the whole build — mirrors `system.py`'s OCR/ASR handling.
    """
    from src.retrieval.index.clip_milvus_index import ClipMilvusIndex

    if isinstance(config, dict):
        model_configs = list(config.get("models") or [])
        default_model_key = config.get("default_model_key")
    else:
        model_configs = list(config)
        default_model_key = None

    milvus_cfg = milvus_cfg or {}
    milvus_host = milvus_cfg.get("host", "localhost")
    milvus_port = milvus_cfg.get("port", 19530)

    indexes: dict[str, FaissIndex] = {}
    for entry in model_configs:
        entry = dict(entry)
        if not entry.get("enabled", True):
            continue
        entry.pop("enabled", None)
        model_key = (
            entry.get("model_key") or entry.get("key") or entry.get("model_name")
        )
        entry.setdefault("model_key", model_key)
        entry.pop("key", None)

        is_milvus = entry.get("backend") == "milvus"
        if is_milvus:
            entry.pop("backend", None)
            for faiss_key in (
                "ef_search",
                "faiss_threads",
                "cache_index_vectors",
                "model_extra",
                "index_path",
            ):
                entry.pop(faiss_key, None)
            entry.setdefault("milvus_host", milvus_host)
            entry.setdefault("milvus_port", milvus_port)

        try:
            indexes[model_key] = (
                ClipMilvusIndex(**entry) if is_milvus else FaissIndex(**entry)
            )
            print(
                f"[IndexManager] loaded model '{model_key}' ({entry.get('model_name')})"
            )
        except Exception as exc:
            print(
                f"[IndexManager] skipping model '{model_key}': {type(exc).__name__}: {exc}"
            )

    return IndexManager(indexes, default_model_key=default_model_key)
