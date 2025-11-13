from __future__ import annotations
import numpy as np

def kmeans_numpy(X: np.ndarray, k: int, iters: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((k, X.shape[1]))
    init_idx = rng.choice(n, size=min(k, n), replace=False)
    C = X[init_idx].copy()
    L = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        L_new = d2.argmin(axis=1).astype(np.int32)
        if np.array_equal(L_new, L):
            break
        L = L_new
        for j in range(k):
            pts = X[L == j]
            if len(pts) > 0:
                C[j] = pts.mean(axis=0)
    return L, C
