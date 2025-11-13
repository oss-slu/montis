from __future__ import annotations
import numpy as np

def slope_from_normals(normals: np.ndarray) -> np.ndarray:
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    return np.degrees(np.arccos(nz)).astype(np.float32)
