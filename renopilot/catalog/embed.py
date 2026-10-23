"""Image and text embeddings in one shared space (open_clip; SigLIP by default).

The model is a config value ("name:pretrained"), so the retrieval eval can compare
CLIP and SigLIP on the same cases instead of asserting one is better.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from renopilot import config


def slug(spec: str) -> str:
    return spec.replace(":", "__").replace("/", "_")


class Embedder:
    def __init__(self, spec: str = config.EMBED_MODEL):
        import open_clip
        import torch

        name, pretrained = spec.split(":")
        self.spec = spec
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    def _norm(self, x) -> np.ndarray:
        x = x.float().cpu().numpy()
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)

    def images(self, paths: list[str | Path], batch: int = 32) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i:i + batch]:
                try:
                    ims.append(self.preprocess(Image.open(config.ROOT / p).convert("RGB")))
                except Exception:                       # unreadable file -> zero vector
                    ims.append(self.torch.zeros_like(ims[-1]) if ims else
                               self.preprocess(Image.new("RGB", (224, 224))))
            with self.torch.no_grad():
                out.append(self._norm(self.model.encode_image(self.torch.stack(ims))))
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            return self._norm(self.model.encode_text(self.tokenizer(texts)))


@lru_cache(maxsize=2)
def get_embedder(spec: str = config.EMBED_MODEL) -> Embedder:
    return Embedder(spec)
