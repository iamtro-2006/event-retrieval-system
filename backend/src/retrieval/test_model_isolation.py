from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from fastapi import HTTPException

from src.api.routers.legacy_search import (
    _require_explicit_model_for_multi_model_search,
    search_api,
)
from src.api.schemas.legacy import SearchRequest, SimilaritySearchRequest
from src.retrieval.retriever.common.orchestrator import QueryPlan
from src.retrieval.retriever.common.scoring import reciprocal_rank_fusion
from src.retrieval.system import RetrievalSystem


class _RecordingOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return pd.DataFrame(), QueryPlan(
            query=query,
            mode=kwargs["mode"],
            use_split=kwargs.get("use_split", True),
            events=[[query]],
        )


@pytest.mark.parametrize(
    ("method", "extra"),
    [
        ("search_semantic", {}),
        ("search_temporal", {}),
        ("search_auto", {}),
    ],
)
def test_every_visual_search_branch_forwards_the_selected_model(method: str, extra: dict) -> None:
    orchestrator = _RecordingOrchestrator()
    system = RetrievalSystem(orchestrator)

    getattr(system, method)("a red car", model_key="model-b", translate=False, **extra)

    assert orchestrator.calls[-1]["model_key"] == "model-b"


def test_legacy_requests_preserve_model_selection() -> None:
    assert SearchRequest(query="car", model_key="model-b").model_key == "model-b"
    assert (
        SimilaritySearchRequest(video_id="video", frame_id=7, model_key="model-b").model_key
        == "model-b"
    )


def test_legacy_unified_endpoint_forwards_model_selection() -> None:
    orchestrator = _RecordingOrchestrator()
    orchestrator.index = SimpleNamespace(model_key="model-a")
    response = asyncio.run(
        search_api(
            payload=SearchRequest(
                query="car",
                model_key="model-b",
                search_mode="semantic",
                use_translate=False,
            ),
            request=SimpleNamespace(),
            system=SimpleNamespace(orchestrator=orchestrator),
            cfg={
                "search": {"default_top_k": 20, "max_top_k": 200, "candidate_multiplier": 5},
                "translate": {"enabled_default": False},
            },
            paths=SimpleNamespace(backend_dir="."),
        )
    )

    assert orchestrator.calls[-1]["model_key"] == "model-b"
    assert response["model_key"] == "model-b"


def test_multi_model_legacy_search_fails_closed_without_model_key() -> None:
    manager = SimpleNamespace(text_search_keys=lambda: ["model-a", "model-b"])
    system = SimpleNamespace(orchestrator=SimpleNamespace(index_manager=manager))

    with pytest.raises(HTTPException) as error:
        _require_explicit_model_for_multi_model_search(
            system, None, feature="semantic search"
        )

    assert error.value.status_code == 400
    _require_explicit_model_for_multi_model_search(
        system, "model-b", feature="semantic search"
    )


def test_retrieval_system_rejects_rows_from_another_model() -> None:
    orchestrator = _RecordingOrchestrator()
    orchestrator.semantic_search = SimpleNamespace(
        _resolve=lambda key: SimpleNamespace(model_key=key)
    )

    def wrong_model_result(query: str, **kwargs):
        return (
            pd.DataFrame([{"model_key": "model-a", "video_id": "v", "keyframe_id": 1}]),
            QueryPlan(query=query, mode=kwargs["mode"], use_split=True, events=[[query]]),
        )

    orchestrator.run_search = wrong_model_result
    system = RetrievalSystem(orchestrator)

    with pytest.raises(RuntimeError, match="isolation violation"):
        system.search_semantic("car", model_key="model-b", translate=False)


def test_rrf_uses_one_shared_identity_contract_across_models() -> None:
    # Older/model-specific metadata may omit ``dataset``. Both rows still
    # identify the same frame and must fuse instead of appearing twice.
    model_a = pd.DataFrame(
        [{"dataset": "set-a", "video_id": "video-1", "keyframe_id": 7, "rank": 1,
          "model_key": "model-a", "search_mode": "semantic"}]
    )
    model_b = pd.DataFrame(
        [{"video_id": "video-1", "keyframe_id": "000007", "rank": 1,
          "model_key": "model-b", "search_mode": "semantic"}]
    )

    result = reciprocal_rank_fusion([model_a, model_b], rrf_k=60)

    assert len(result) == 1
    assert result.iloc[0]["matched_sources"] == 2
    assert result.iloc[0]["source_models"] == [
        "semantic / model-a",
        "semantic / model-b",
    ]


def test_rrf_does_not_count_a_duplicate_twice_within_one_model() -> None:
    duplicated = pd.DataFrame(
        [
            {"video_id": "video-1", "keyframe_id": 7, "rank": 1, "model_key": "model-a"},
            {"video_id": "video-1", "keyframe_id": 7, "rank": 2, "model_key": "model-a"},
        ]
    )

    result = reciprocal_rank_fusion([duplicated], rrf_k=60)

    assert len(result) == 1
    assert result.iloc[0]["matched_sources"] == 1
    assert result.iloc[0]["rrf_score"] == pytest.approx(1.0 / 61.0)
