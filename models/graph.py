from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
from scipy.spatial import cKDTree

def build_graph(count: int,
                indices: Optional[np.ndarray],
                P: Optional[np.ndarray] = None,
                knn: int = 16,
                max_nodes_knn: int = 300_000) -> Optional[List[List[int]]]:
    if indices is not None:
        G: List[List[int]] = [[] for _ in range(count)]
        tris = indices.reshape(-1, 3)
        for a, b, c in tris:
            G[a].extend([b, c]); G[b].extend([a, c]); G[c].extend([a, b])
        for i in range(count):
            G[i] = sorted(set(G[i]))
        return G

    if P is None or count == 0:
        return None
    if count > max_nodes_knn:
        logging.getLogger(__name__).warning("kNN graph skipped: %d > %d", count, max_nodes_knn)
        return None

    tree = cKDTree(P)
    kq = min(knn + 1, count)
    nbrs = tree.query(P, k=kq)[1]
    if nbrs.ndim == 1:
        nbrs = nbrs.reshape(-1, 1)
    nbrs = nbrs[:, 1:]
    return [sorted(set(row.tolist())) for row in nbrs]
