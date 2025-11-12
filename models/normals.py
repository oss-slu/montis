from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def estimate_normals(P: np.ndarray, k: int = 16) -> np.ndarray:
    tree = cKDTree(P)
    normals = np.empty_like(P, dtype=np.float32)
    k = min(k, len(P))
    for i in range(P.shape[0]):
        _, idx = tree.query(P[i], k=k)
        pts = P[idx]
        C = np.cov(pts.T)
        _, v = np.linalg.eigh(C)
        n = v[:, 0]
        if n[2] < 0: n = -n
        normals[i] = n
    return normals
