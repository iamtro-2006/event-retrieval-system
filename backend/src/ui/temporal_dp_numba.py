"""
Numba-accelerated DP kernels for temporal search.

All functions here are pure-numeric (numpy arrays in, scalars/arrays out)
so Numba can compile them in nopython mode.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList


# ---------------------------------------------------------------------------
# Core DP on a single score matrix  (replaces _run_dp_on_window)
# ---------------------------------------------------------------------------
@njit(cache=True)
def run_dp_on_window(S: np.ndarray):
    """
    Strict temporal-order DP on score matrix S[m, n].
    Returns (best_score, path_array, path_len).
    path_array is pre-allocated to length m; only [:path_len] is valid.
    """
    m, n = S.shape
    NEG_INF = np.float32(-np.inf)

    dp = np.full((m, n), NEG_INF, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)

    # Base case: first query row
    for j in range(n):
        dp[0, j] = S[0, j]

    # Fill DP table
    for qi in range(1, m):
        best_prev_score = NEG_INF
        best_prev_idx = np.int32(-1)

        for fj in range(n):
            # Update best predecessor from dp[qi-1, 0..fj-1]
            if fj > 0:
                prev_val = dp[qi - 1, fj - 1]
                if prev_val > best_prev_score:
                    best_prev_score = prev_val
                    best_prev_idx = np.int32(fj - 1)

            # Assign dp value if we have a valid predecessor
            if best_prev_idx >= 0 and math.isfinite(best_prev_score):
                dp[qi, fj] = best_prev_score + S[qi, fj]
                parent[qi, fj] = best_prev_idx

    # Find best ending position for last query
    best_score = NEG_INF
    last_idx = np.int32(0)
    for j in range(n):
        if dp[m - 1, j] > best_score:
            best_score = dp[m - 1, j]
            last_idx = np.int32(j)

    if not math.isfinite(best_score):
        return NEG_INF, np.empty(0, dtype=np.int32), np.int32(0)

    # Backtrack path
    path = np.empty(m, dtype=np.int32)
    path[m - 1] = last_idx
    cur = last_idx

    for qi in range(m - 1, 0, -1):
        cur = parent[qi, cur]
        if cur < 0:
            return NEG_INF, np.empty(0, dtype=np.int32), np.int32(0)
        path[qi - 1] = cur

    return best_score, path, np.int32(m)


# ---------------------------------------------------------------------------
# Near-tied suffix detection  (replaces _has_near_tied_suffix_choices)
# ---------------------------------------------------------------------------
@njit(cache=True)
def has_near_tied_suffix_choices(S: np.ndarray, atol: float = 1e-6) -> bool:
    m, n = S.shape
    NEG_INF = np.float32(-np.inf)

    dp = np.full((m, n), NEG_INF, dtype=np.float32)
    for j in range(n):
        dp[m - 1, j] = S[m - 1, j]

    for qi in range(m - 2, -1, -1):
        suffix_scores = np.full(n, NEG_INF, dtype=np.float32)
        best_score = NEG_INF

        for fj in range(n - 1, -1, -1):
            suffix_scores[fj] = best_score
            candidate = dp[qi + 1, fj]

            if (
                math.isfinite(candidate)
                and math.isfinite(best_score)
                and abs(candidate - best_score) <= atol
            ):
                return True

            if candidate >= best_score:
                best_score = candidate

        for fj in range(n):
            if math.isfinite(suffix_scores[fj]):
                dp[qi, fj] = S[qi, fj] + suffix_scores[fj]

    best_score = NEG_INF
    for start in range(n - 1, -1, -1):
        candidate = dp[0, start]
        if (
            math.isfinite(candidate)
            and math.isfinite(best_score)
            and abs(candidate - best_score) <= atol
        ):
            return True
        if candidate >= best_score:
            best_score = candidate

    return False


# ---------------------------------------------------------------------------
# Window-based temporal candidates  (replaces _window_temporal_candidates)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _searchsorted_right(arr, value):
    """Binary search for rightmost insertion point (equivalent to np.searchsorted side='right')."""
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def window_temporal_candidates(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float,
):
    """
    Returns parallel arrays: (scores, paths_flat, path_offsets, start_times, end_times, count).
    paths_flat contains all path indices concatenated.
    path_offsets[i] is the start offset in paths_flat for candidate i.
    """
    m, n = S.shape

    # Pre-allocate generous buffers
    max_candidates = n
    scores = np.empty(max_candidates, dtype=np.float32)
    start_times = np.empty(max_candidates, dtype=np.float32)
    end_times = np.empty(max_candidates, dtype=np.float32)
    paths_flat = np.empty(max_candidates * m, dtype=np.int32)
    path_offsets = np.empty(max_candidates, dtype=np.int32)
    count = 0

    for start in range(n):
        if duration_limit == -1.0:
            end = n
        else:
            end = _searchsorted_right(timestamps, timestamps[start] + duration_limit)

        if end - start < m:
            continue

        local_S = S[:, start:end]
        local_score, local_path, path_len = run_dp_on_window(local_S)

        if path_len == 0 or not math.isfinite(local_score):
            continue

        # Store candidate
        offset = count * m
        path_offsets[count] = offset
        scores[count] = local_score

        for pi in range(m):
            paths_flat[offset + pi] = start + local_path[pi]

        start_times[count] = timestamps[paths_flat[offset]]
        end_times[count] = timestamps[paths_flat[offset + m - 1]]
        count += 1

    return (
        scores[:count],
        paths_flat[: count * m],
        path_offsets[:count],
        start_times[:count],
        end_times[:count],
        np.int32(m),
    )


# ---------------------------------------------------------------------------
# Suffix-DP temporal candidates  (replaces _suffix_temporal_candidates)
# ---------------------------------------------------------------------------
@njit(cache=True)
def suffix_temporal_candidates(S: np.ndarray, timestamps: np.ndarray):
    m, n = S.shape
    NEG_INF = np.float32(-np.inf)

    dp = np.full((m, n), NEG_INF, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)

    for j in range(n):
        dp[m - 1, j] = S[m - 1, j]

    for qi in range(m - 2, -1, -1):
        suffix_scores = np.full(n, NEG_INF, dtype=np.float32)
        suffix_indices = np.full(n, -1, dtype=np.int32)
        best_score = NEG_INF
        best_idx = np.int32(-1)

        for fj in range(n - 1, -1, -1):
            suffix_scores[fj] = best_score
            suffix_indices[fj] = best_idx

            if dp[qi + 1, fj] >= best_score:
                best_score = dp[qi + 1, fj]
                best_idx = np.int32(fj)

        for fj in range(n):
            if math.isfinite(suffix_scores[fj]):
                dp[qi, fj] = S[qi, fj] + suffix_scores[fj]
                parent[qi, fj] = suffix_indices[fj]

    # Build suffix-best arrays
    suffix_best_scores = np.full(n, NEG_INF, dtype=np.float32)
    suffix_best_indices = np.full(n, -1, dtype=np.int32)
    best_score = NEG_INF
    best_idx = np.int32(-1)

    for start in range(n - 1, -1, -1):
        if dp[0, start] >= best_score:
            best_score = dp[0, start]
            best_idx = np.int32(start)
        suffix_best_scores[start] = best_score
        suffix_best_indices[start] = best_idx

    # Collect candidates
    max_candidates = n
    scores = np.empty(max_candidates, dtype=np.float32)
    start_times = np.empty(max_candidates, dtype=np.float32)
    end_times = np.empty(max_candidates, dtype=np.float32)
    paths_flat = np.empty(max_candidates * m, dtype=np.int32)
    path_offsets = np.empty(max_candidates, dtype=np.int32)
    count = 0

    for start in range(n):
        score = suffix_best_scores[start]
        frame_idx = suffix_best_indices[start]

        if frame_idx < 0 or not math.isfinite(score):
            continue

        # Trace path
        path = np.empty(m, dtype=np.int32)
        path[0] = frame_idx
        valid = True

        for qi in range(m - 1):
            frame_idx = parent[qi, frame_idx]
            if frame_idx < 0:
                valid = False
                break
            path[qi + 1] = frame_idx

        if not valid:
            continue

        offset = count * m
        path_offsets[count] = offset
        scores[count] = score

        for pi in range(m):
            paths_flat[offset + pi] = path[pi]

        start_times[count] = timestamps[path[0]]
        end_times[count] = timestamps[path[m - 1]]
        count += 1

    return (
        scores[:count],
        paths_flat[: count * m],
        path_offsets[:count],
        start_times[:count],
        end_times[:count],
        np.int32(m),
    )


# ---------------------------------------------------------------------------
# NMS selection  (replaces _select_non_overlapping_candidates)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _sequence_overlap(path_a, path_b):
    """Overlap ratio between two index arrays."""
    set_count = 0
    min_len = min(len(path_a), len(path_b))
    if min_len == 0:
        return 0.0

    for i in range(len(path_a)):
        for j in range(len(path_b)):
            if path_a[i] == path_b[j]:
                set_count += 1
                break

    return set_count / min_len


@njit(cache=True)
def _time_iou_jit(sa, ea, sb, eb):
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    union = max(ea, eb) - min(sa, sb)
    if union <= 1e-12:
        return 0.0
    return inter / union


@njit(cache=True)
def select_non_overlapping(
    scores,
    paths_flat,
    path_offsets,
    start_times,
    end_times,
    path_len,
    max_sequences,
    overlap_threshold,
):
    """
    NMS-style selection. Returns indices into the candidate arrays.
    Candidates must already be sorted by score descending.
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    # Sort indices by score descending
    order = np.argsort(-scores)

    selected = np.empty(min(n, max_sequences), dtype=np.int32)
    sel_count = 0

    for oi in range(n):
        idx = order[oi]
        offset = path_offsets[idx]
        path_i = paths_flat[offset: offset + path_len]
        st_i = start_times[idx]
        et_i = end_times[idx]

        duplicated = False
        for si in range(sel_count):
            sel_idx = selected[si]
            sel_offset = path_offsets[sel_idx]
            path_s = paths_flat[sel_offset: sel_offset + path_len]

            p_overlap = _sequence_overlap(path_i, path_s)
            t_overlap = _time_iou_jit(st_i, et_i, start_times[sel_idx], end_times[sel_idx])

            if p_overlap >= overlap_threshold or t_overlap >= overlap_threshold:
                duplicated = True
                break

        if not duplicated:
            selected[sel_count] = idx
            sel_count += 1
            if sel_count >= max_sequences:
                break

    return selected[:sel_count]
