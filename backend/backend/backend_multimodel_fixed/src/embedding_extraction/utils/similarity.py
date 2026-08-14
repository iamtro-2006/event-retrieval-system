"""Cosine-similarity search over saved embeddings.

Mirrors the pattern from the LAVIS "image and text features extraction"
tutorial: normalize each embedding to unit length, then a plain matrix
multiply gives cosine similarity (since for unit vectors, dot product ==
cosine similarity):

    images_embedding /= images_embedding.norm(dim=-1, keepdim=True)
    texts_embedding  /= texts_embedding.norm(dim=-1, keepdim=True)
    similarity_images_texts = images_embedding @ texts_embedding.T

The image side embeddings here are the ones already saved to .npy by
`encode_keyframe_images()` (backend blip2.py with pooling="first" ->
`image_embeds_proj[:, 0, :]`, the trained ITC projection space, one flat
256-dim vector per keyframe). This module only handles the matching step;
producing the text-side embedding (same projection space, same
`[:, 0, :]` convention) is a separate step done at query time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis. No-op if already unit-norm
    (e.g. blip2's image_embeds_proj is normalized inside the model already),
    but always safe/idempotent to call again before a similarity search.
    """
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.clip(norm, a_min=1e-12, a_max=None)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of `a` and every row of `b`.

    a: (n, dim), b: (m, dim) -> (n, m). Same operation as
    `images_embedding @ texts_embedding.T` in the tutorial, just normalizing
    both sides first so the caller doesn't have to remember to.
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    return a_norm @ b_norm.T


def load_gallery(embedding_dir: Path) -> tuple[np.ndarray, list[Path]]:
    """Load every .npy embedding under `embedding_dir` (recursively) into
    one (n, dim) array, keeping track of which file each row came from -
    the equivalent of the tutorial's `images_embedding = torch.cat(...)`
    accumulation loop, but reading back what the pipeline already saved to
    disk instead of holding everything in memory during extraction.
    """
    paths = sorted(Path(embedding_dir).rglob("*.npy"))
    if not paths:
        return np.empty((0, 0), dtype=np.float32), []

    vectors = [np.load(p).astype(np.float32) for p in paths]
    return np.vstack(vectors), paths


def search_text_query(
    text_embedding: np.ndarray,
    embedding_dir: Path,
    top_k: int = 10,
) -> list[tuple[Path, float]]:
    """Image-text retrieval: rank saved keyframe image embeddings against
    ONE text query embedding (shape (dim,) or (1, dim), already extracted
    the same way - same backend/checkpoint, same "[:, 0, :]" projected
    single-vector convention as the image side).

    Returns the top_k (image_path, cosine_similarity) pairs, best first.
    """
    gallery, paths = load_gallery(embedding_dir)
    if gallery.size == 0:
        return []

    query = np.asarray(text_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity_matrix(query, gallery)[0]  # (n,)

    top_k = min(top_k, len(paths))
    top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(paths[i], float(sims[i])) for i in top_idx]
