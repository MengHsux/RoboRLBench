"""Running statistics used for observation / state / reward normalisation."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

__all__ = ["RunningMeanStd"]


class RunningMeanStd:
    """Parallel (Chan et al.) running mean / variance estimator.

    Works with numpy arrays (rollout collection) and torch tensors (updates);
    ``normalize`` returns the same type it receives.
    """

    def __init__(self, shape: Union[int, tuple] = (), eps: float = 1e-4) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.eps = float(eps)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(eps)

    # ------------------------------------------------------------------ #
    def update(self, x: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.shape) else 1

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count
        m2 += delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = np.maximum(m2 / total, 1e-12)
        self.count = total

    # ------------------------------------------------------------------ #
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(self, x, clip: float = 10.0):
        mean = self.mean.astype(np.float32)
        std = self.std.astype(np.float32)
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(std, dtype=x.dtype, device=x.device)
            out = (x - m) / (s + 1e-8)
            return torch.clamp(out, -clip, clip)
        out = (np.asarray(x, dtype=np.float32) - mean) / (std + 1e-8)
        return np.clip(out, -clip, clip).astype(np.float32)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count,
                "shape": self.shape, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
