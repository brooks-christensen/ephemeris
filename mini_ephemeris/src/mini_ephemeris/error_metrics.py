from __future__ import annotations
import numpy as np


def position_error(model_pos: np.ndarray,
                   truth_pos: np.ndarray) -> np.ndarray:
    diff = model_pos - truth_pos
    return np.linalg.norm(diff, axis=-1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))