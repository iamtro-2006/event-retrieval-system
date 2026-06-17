from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.ui.temporal_search import (  # noqa: E402
    _run_dp_on_window,
    _sequence_overlap_ratio,
    _temporal_topk_dp,
    _time_iou,
)


def legacy_run_dp_on_window(S: np.ndarray) -> tuple[float, list[int]]:
    m, n = S.shape
    dp = np.full((m, n), -np.inf, dtype=np.float32)
    parent = np.full((m, n), -1, dtype=np.int32)
    dp[0] = S[0]

    for qi in range(1, m):
        best_prev_score = -np.inf
        best_prev_idx = -1

        for fj in range(n):
            if fj > 0 and dp[qi - 1, fj - 1] > best_prev_score:
                best_prev_score = dp[qi - 1, fj - 1]
                best_prev_idx = fj - 1

            if best_prev_idx != -1:
                dp[qi, fj] = best_prev_score + S[qi, fj]
                parent[qi, fj] = best_prev_idx

    last_idx = int(np.argmax(dp[m - 1]))
    best_score = float(dp[m - 1, last_idx])

    if not np.isfinite(best_score):
        return -np.inf, []

    path = [last_idx]

    for qi in range(m - 1, 0, -1):
        last_idx = int(parent[qi, last_idx])

        if last_idx < 0:
            return -np.inf, []

        path.append(last_idx)

    path.reverse()
    return best_score, path


def legacy_temporal_topk_dp(
    S: np.ndarray,
    timestamps: np.ndarray,
    duration_limit: float = -1,
    max_sequences: int = 3,
    overlap_threshold: float = 0.6,
) -> list[tuple[float, list[int]]]:
    m, n = S.shape

    if n < m:
        return []

    candidates: list[tuple[float, list[int], float, float]] = []

    for start in range(n):
        if duration_limit == -1:
            end = n
        else:
            end = int(
                np.searchsorted(
                    timestamps,
                    timestamps[start] + duration_limit,
                    side="right",
                )
            )

        if end - start < m:
            continue

        local_score, local_path = legacy_run_dp_on_window(S[:, start:end])

        if not local_path or not np.isfinite(local_score):
            continue

        global_path = [start + idx for idx in local_path]
        start_time = float(timestamps[global_path[0]])
        end_time = float(timestamps[global_path[-1]])
        candidates.append((float(local_score), global_path, start_time, end_time))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: list[tuple[float, list[int], float, float]] = []

    for score, path, start_time, end_time in candidates:
        duplicated = False

        for _, selected_path, selected_start, selected_end in selected:
            path_overlap = _sequence_overlap_ratio(path, selected_path)
            time_overlap = _time_iou(
                start_time,
                end_time,
                selected_start,
                selected_end,
            )

            if path_overlap >= overlap_threshold or time_overlap >= overlap_threshold:
                duplicated = True
                break

        if duplicated:
            continue

        selected.append((score, path, start_time, end_time))

        if len(selected) >= max_sequences:
            break

    return [(score, path) for score, path, _, _ in selected]


def assert_same_results(
    legacy: list[tuple[float, list[int]]],
    current: list[tuple[float, list[int]]],
    *,
    case_name: str,
) -> None:
    if len(legacy) != len(current):
        raise AssertionError(
            f"{case_name}: result count changed: legacy={len(legacy)} current={len(current)}"
        )

    for rank, ((legacy_score, legacy_path), (current_score, current_path)) in enumerate(
        zip(legacy, current),
        start=1,
    ):
        if legacy_path != current_path:
            raise AssertionError(
                f"{case_name}: rank {rank} path changed: "
                f"legacy={legacy_path} current={current_path}"
            )

        if not np.isclose(legacy_score, current_score, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                f"{case_name}: rank {rank} score changed: "
                f"legacy={legacy_score} current={current_score}"
            )


def make_timestamps(rng: np.random.Generator, n: int, irregular: bool) -> np.ndarray:
    if not irregular:
        return np.arange(n, dtype=np.float32)

    increments = rng.uniform(0.03, 3.0, size=n).astype(np.float32)
    return np.cumsum(increments, dtype=np.float32)


def verify_case(
    S: np.ndarray,
    timestamps: np.ndarray,
    *,
    duration_limit: float,
    max_sequences: int,
    overlap_threshold: float,
    case_name: str,
) -> None:
    legacy_window = legacy_run_dp_on_window(S)
    current_window = _run_dp_on_window(S)
    assert_same_results([legacy_window], [current_window], case_name=f"{case_name}/window")

    legacy_topk = legacy_temporal_topk_dp(
        S,
        timestamps,
        duration_limit=duration_limit,
        max_sequences=max_sequences,
        overlap_threshold=overlap_threshold,
    )
    current_topk = _temporal_topk_dp(
        S,
        timestamps,
        duration_limit=duration_limit,
        max_sequences=max_sequences,
        overlap_threshold=overlap_threshold,
    )
    assert_same_results(legacy_topk, current_topk, case_name=f"{case_name}/topk")


def run_regression(seed: int, cases: int) -> None:
    rng = np.random.default_rng(seed)

    fixed_cases = [
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.3, 0.2], [0.1, 0.4]], dtype=np.float32),
        np.full((2, 4), 0.5, dtype=np.float32),
        np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float32),
    ]

    checked = 0

    for idx, S in enumerate(fixed_cases):
        timestamps = np.arange(S.shape[1], dtype=np.float32)
        verify_case(
            S,
            timestamps,
            duration_limit=-1,
            max_sequences=3,
            overlap_threshold=0.6,
            case_name=f"fixed-{idx}",
        )
        checked += 1

    for idx in range(cases):
        m = int(rng.integers(1, 8))
        n = int(rng.integers(m, 80))
        S = rng.normal(size=(m, n)).astype(np.float32)

        if idx % 11 == 0:
            S = np.round(S, 1).astype(np.float32)

        timestamps = make_timestamps(rng, n, irregular=bool(idx % 3 == 0))
        duration_limit = -1 if idx % 2 == 0 else float(rng.uniform(0.2, max(1.0, n / 3)))
        max_sequences = int(rng.integers(1, 6))
        overlap_threshold = float(rng.uniform(0.3, 0.9))

        verify_case(
            S,
            timestamps,
            duration_limit=duration_limit,
            max_sequences=max_sequences,
            overlap_threshold=overlap_threshold,
            case_name=f"random-{idx}",
        )
        checked += 1

    print(
        "Temporal search regression passed: "
        f"{checked} cases, seed={seed}, legacy outputs preserved."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify temporal search optimization preserves legacy scores and paths."
    )
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()

    run_regression(seed=args.seed, cases=args.cases)


if __name__ == "__main__":
    main()