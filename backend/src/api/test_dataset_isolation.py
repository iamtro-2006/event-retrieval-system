import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.legacy.deps import get_dataset_resources
from src.api.legacy.serializers import dict_to_result_FAST, resolve_keyframe_path_from_dict
from src.api.routers.legacy_search import get_frame_info, get_surrounding_frames, get_video_preview
from src.api.routers.search import get_retrieval_system, list_available_models


def _request():
    aic_system = object()
    cam_system = object()
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        retrieval_systems={"aic": aic_system, "cam": cam_system},
        dataset_configs={"aic": {"name": "aic"}, "cam": {"name": "cam"}},
        dataset_paths={"aic": "aic-paths", "cam": "cam-paths"},
        dataset_errors={"aic": None, "cam": None},
    ))), aic_system, cam_system


def test_dataset_resources_select_distinct_physical_systems():
    request, aic_system, cam_system = _request()
    assert get_dataset_resources(request, "aic") == (aic_system, {"name": "aic"}, "aic-paths")
    assert get_dataset_resources(request, "CAM") == (cam_system, {"name": "cam"}, "cam-paths")
    assert aic_system is not cam_system


def test_unavailable_dataset_never_falls_back_to_other_index():
    request, aic_system, _ = _request()
    request.app.state.retrieval_systems["cam"] = None
    request.app.state.dataset_errors["cam"] = "missing CAM index"
    with pytest.raises(HTTPException) as exc_info:
        get_dataset_resources(request, "cam")
    assert exc_info.value.status_code == 503
    assert "CAM" in exc_info.value.detail
    assert get_dataset_resources(request, "aic")[0] is aic_system


def test_model_list_comes_from_requested_dataset_only():
    class StubSystem:
        def __init__(self, models):
            self._models = models

        def available_models(self):
            return self._models

    aic_system = StubSystem(["pe-core"])
    cam_system = StubSystem(["cam-model"])
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        retrieval_systems={"aic": aic_system, "cam": cam_system},
        dataset_errors={"aic": None, "cam": None},
    )))

    assert get_retrieval_system(request, "aic") is aic_system
    assert list_available_models(request, "aic").models == ["pe-core"]
    assert list_available_models(request, "cam").models == ["cam-model"]


def test_post_search_helpers_read_only_the_requested_dataset():
    def make_system(dataset):
        metadata = pd.DataFrame([
            {"dataset": dataset, "video_id": "shared-video", "keyframe_id": 1, "frame_idx": 10, "timestamp_sec": 1.0},
            {"dataset": dataset, "video_id": "shared-video", "keyframe_id": 2, "frame_idx": 20, "timestamp_sec": 2.0},
            {"dataset": dataset, "video_id": "shared-video", "keyframe_id": 3, "frame_idx": 30, "timestamp_sec": 3.0},
        ])
        index = SimpleNamespace(
            metadata=metadata,
            _rows_by_video={"shared-video": [0, 1, 2]},
            _row_by_video_frame={("shared-video", 1): 0, ("shared-video", 2): 1, ("shared-video", 3): 2},
        )
        return SimpleNamespace(orchestrator=SimpleNamespace(index=index))

    aic_paths = SimpleNamespace(backend_dir=Path("."), keyframes_root=Path("aic"), dataset_key="aic")
    cam_paths = SimpleNamespace(backend_dir=Path("."), keyframes_root=Path("cam"), dataset_key="cam")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        retrieval_systems={"aic": make_system("aic"), "cam": make_system("cam")},
        dataset_configs={"aic": {}, "cam": {}},
        dataset_paths={"aic": aic_paths, "cam": cam_paths},
        dataset_errors={"aic": None, "cam": None},
    )))

    frame = asyncio.run(get_frame_info("shared-video", 2, "cam", request))
    preview = asyncio.run(get_video_preview("shared-video", "cam", request, frame_id=20))
    surround = asyncio.run(get_surrounding_frames("shared-video", 2, "cam", request, radius=1))

    assert frame["raw"]["dataset"] == "cam"
    assert preview["raw"]["dataset"] == "cam"
    assert [item["raw"]["dataset"] for item in surround["frames"]] == ["cam", "cam", "cam"]


@pytest.mark.parametrize(("actual_suffix", "metadata_suffix"), [(".webp", ".webp"), (".jpg", ".webp")])
def test_keyframe_resolver_uses_the_file_that_actually_exists(tmp_path, actual_suffix, metadata_suffix):
    keyframes_root = tmp_path / "keyframes"
    video_dir = keyframes_root / "N001" / "N001-V001"
    video_dir.mkdir(parents=True)
    actual_file = video_dir / f"000000{actual_suffix}"
    actual_file.write_bytes(b"image-placeholder")
    item = {
        "dataset": "N001",
        "video_id": "N001-V001",
        "keyframe_id": "000000",
        "keyframe_path": f"../old/keyframes/N001/N001-V001/000000{metadata_suffix}",
    }

    resolved = resolve_keyframe_path_from_dict(item, keyframes_root, tmp_path)
    serialized = dict_to_result_FAST(item, keyframes_root, tmp_path, "cam")

    assert resolved.replace("\\", "/").endswith(f"/N001/N001-V001/000000{actual_suffix}")
    assert serialized["image_rel_path"] == f"N001/N001-V001/000000{actual_suffix}"
    assert serialized["image_url"].endswith(f"/N001/N001-V001/000000{actual_suffix}")


def test_keyframe_resolver_preserves_video_group_when_dataset_is_collection(tmp_path):
    keyframes_root = tmp_path / "keyframes"
    video_dir = keyframes_root / "L21" / "L21_V001"
    video_dir.mkdir(parents=True)
    (video_dir / "000029.jpg").write_bytes(b"image-placeholder")
    item = {
        "dataset": "aic",
        "video_id": "L21_V001",
        "keyframe_id": "000029",
        "keyframe_path": "../old/keyframes/000029.jpg",
    }

    resolved = resolve_keyframe_path_from_dict(item, keyframes_root, tmp_path)
    serialized = dict_to_result_FAST(item, keyframes_root, tmp_path, "aic")

    assert resolved.replace("\\", "/").endswith("/L21/L21_V001/000029.jpg")
    assert serialized["image_rel_path"] == "L21/L21_V001/000029.jpg"


def test_keyframe_resolver_finds_partitioned_split_video(tmp_path):
    keyframes_root = tmp_path / "keyframes"
    video_dir = keyframes_root / "L26_d" / "L26_V383"
    video_dir.mkdir(parents=True)
    (video_dir / "000041.jpg").write_bytes(b"image-placeholder")
    item = {
        "dataset": "L26",
        "video_id": "L26_V383",
        "keyframe_id": "000041",
        "keyframe_path": "../old/keyframes/L26/L26_V383/000041.jpg",
    }

    resolved = resolve_keyframe_path_from_dict(item, keyframes_root, tmp_path)
    serialized = dict_to_result_FAST(item, keyframes_root, tmp_path, "aic")

    assert resolved.replace("\\", "/").endswith("/L26_d/L26_V383/000041.jpg")
    assert serialized["image_rel_path"] == "L26_d/L26_V383/000041.jpg"


def test_result_resolves_partitioned_aic_video_and_map_metadata(tmp_path):
    dataset_root = tmp_path / "AIC"
    keyframes_root = dataset_root / "keyframes"
    video_id = "L26_V087"
    frame_dir = keyframes_root / "L26_a" / video_id
    frame_dir.mkdir(parents=True)
    (frame_dir / "000001.jpg").write_bytes(b"image-placeholder")
    video_path = dataset_root / "videos" / "L26_a" / f"{video_id}.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video-placeholder")
    map_path = dataset_root / "map_keyframes" / "L26_a" / f"{video_id}.csv"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        "keyframe_id,video_id,frame_idx,timestamp_sec,fps\n1,L26_V087,21,0.84,25.0\n",
        encoding="utf-8",
    )
    item = {
        "dataset": "L26",
        "video_id": video_id,
        "keyframe_id": "000001",
        "keyframe_id_int": 1,
        "frame_idx": float("nan"),
        "timestamp_sec": float("nan"),
        "fps": float("nan"),
        "keyframe_path": f"../old/keyframes/L26/{video_id}/000001.jpg",
    }

    result = dict_to_result_FAST(item, keyframes_root, tmp_path, "aic")

    assert result["video_url"] == f"/static/aic/videos/L26_a/{video_id}.mp4"
    assert result["map_url"] == f"/static/aic/map-keyframes/L26_a/{video_id}.csv"
    assert result["frame_idx"] == 21
    assert result["fps"] == 25.0
    assert result["timestamp"] == 0.84
