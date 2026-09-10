from __future__ import annotations

import sys
import types

import pytest
import torch

from src.embedding_extraction.models.backends import perception_encoder
from src.embedding_extraction.models.registry import MODEL_PRESETS, load_model


class _FakeClip(torch.nn.Module):
    image_size = 336
    context_length = 32
    visual = types.SimpleNamespace(output_dim=4)
    last_from_config = None

    @classmethod
    def available_configs(cls):
        return ["PE-Core-L14-336"]

    @classmethod
    def from_config(cls, name, pretrained=False, checkpoint_path=None):
        cls.last_from_config = (name, pretrained, checkpoint_path)
        return cls()

    def encode_image(self, batch):
        return batch.mean(dim=(1, 2, 3), keepdim=False)[:, None].repeat(1, 4)

    def encode_text(self, tokens):
        return tokens.float().mean(dim=1, keepdim=True).repeat(1, 4)


def _install_fake_package(monkeypatch):
    core = types.ModuleType("core")
    core.__path__ = []
    vision = types.ModuleType("core.vision_encoder")
    vision.__path__ = []
    pe = types.ModuleType("core.vision_encoder.pe")
    pe.CLIP = _FakeClip
    transforms = types.ModuleType("core.vision_encoder.transforms")
    transforms.get_image_transform = lambda size, center_crop=False: (
        lambda image: torch.ones(3, 2, 2)
    )
    transforms.get_text_tokenizer = lambda context_length: (
        lambda texts: torch.ones(len(texts), context_length, dtype=torch.long)
    )
    core.vision_encoder = vision
    vision.pe = pe
    vision.transforms = transforms
    for name, module in {
        "core": core,
        "core.vision_encoder": vision,
        "core.vision_encoder.pe": pe,
        "core.vision_encoder.transforms": transforms,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_all_pe_core_sizes_are_registered():
    for size in ("T16-384", "S16-384", "B16-224", "L14-336", "G14-448"):
        assert MODEL_PRESETS[f"PE-Core-{size}"]["backend"] == "perception_encoder"
        assert MODEL_PRESETS[f"facebook/PE-Core-{size}"]["model_name"] == f"PE-Core-{size}"


def test_loads_hf_alias_and_encodes_both_modalities(monkeypatch):
    _install_fake_package(monkeypatch)

    loaded = perception_encoder.load(
        "facebook/PE-Core-L14-336",
        pretrained="hf",
        device_name="cpu",
        precision="amp",
    )

    assert _FakeClip.last_from_config == ("PE-Core-L14-336", True, None)
    assert loaded.precision == "fp32"
    assert loaded.embedding_dim == 4
    assert loaded.supports_text is True
    assert loaded.model.encode_image(torch.ones(2, 3, 2, 2)).shape == (2, 4)
    assert loaded.model.encode_text(["one", "two"]).shape == (2, 4)


def test_registry_dispatches_pe_core_preset(monkeypatch):
    _install_fake_package(monkeypatch)
    loaded = load_model("PE-Core-L14-336", device_name="cpu")
    assert loaded.supports_text is True
    assert _FakeClip.last_from_config == ("PE-Core-L14-336", True, None)


def test_local_checkpoint_override(monkeypatch, tmp_path):
    _install_fake_package(monkeypatch)
    checkpoint = tmp_path / "pe.pt"

    perception_encoder.load(
        "PE-Core-L14-336",
        pretrained=False,
        device_name="cpu",
        perception_encoder={"checkpoint_path": str(checkpoint)},
    )

    assert _FakeClip.last_from_config == (
        "PE-Core-L14-336",
        True,
        str(checkpoint),
    )


def test_hugging_face_repo_id_uses_default_checkpoint():
    assert perception_encoder._resolve_pretrained(
        "facebook/PE-Core-L14-336", None
    ) == (True, None)


def test_rejects_non_clip_pe_config(monkeypatch):
    _install_fake_package(monkeypatch)
    with pytest.raises(ValueError, match="not a PE-Core CLIP config"):
        perception_encoder.load("PE-Spatial-L14-448", device_name="cpu")
