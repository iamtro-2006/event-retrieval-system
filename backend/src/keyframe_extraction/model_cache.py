"""Worker-owned model lifetimes, independent of per-video output/configuration."""
from __future__ import annotations

import gc


class KeyframeModels:
    def __init__(self):
        self._omni = None
        self._omni_key = None
        self._embedding = None
        self._embedding_key = None

    def omni(self, cfg, loader):
        key = (str(cfg.weights_path), str(cfg.cache_dir))
        if self._omni is None:
            model = loader(cfg)
            self._omni, self._omni_key = model, key
        elif self._omni_key != key:
            raise ValueError('OmniShotCut weights changed; start a new worker')
        return self._omni

    def embedding(self, cfg, loader, logger):
        key = (cfg.model_name, cfg.pretrained, cfg.precision, cfg.device, cfg.backend)
        if self._embedding is None:
            loaded = loader(model_name=cfg.model_name, pretrained=cfg.pretrained,
                            precision=cfg.precision, device_name=cfg.device,
                            backend=cfg.backend, logger=logger)
            self._embedding, self._embedding_key = loaded, key
        elif self._embedding_key != key:
            raise ValueError('Embedding model configuration changed; start a new worker')
        return self._embedding

    def close(self):
        self._omni = self._embedding = None
        self._omni_key = self._embedding_key = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
