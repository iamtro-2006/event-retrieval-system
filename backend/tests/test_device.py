from __future__ import annotations

import pytest


class TestResolveDevice:
    def test_auto_returns_cpu_when_no_cuda(self, monkeypatch):
        from src.utils.device import resolve_device

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        monkeypatch.setattr("torch.cuda", FakeCuda)
        dev = resolve_device("auto")
        assert str(dev) == "cpu"

    def test_cuda_falls_back_to_cpu_when_unavailable(self, monkeypatch):
        from src.utils.device import resolve_device

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        monkeypatch.setattr("torch.cuda", FakeCuda)
        dev = resolve_device("cuda:0")
        assert str(dev) == "cpu"

    def test_cpu_passthrough(self):
        from src.utils.device import resolve_device

        dev = resolve_device("cpu")
        assert str(dev) == "cpu"
