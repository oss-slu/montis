from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def snap_to_feasible(P: np.ndarray, S: np.ndarray, ix: int, maxSlope: float) -> int:
    if 0 <= ix < len(S) and S[ix] <= maxSlope:
        return int(ix)
    good = np.where(S <= maxSlope)[0]
    if good.size == 0:
        return int(ix)
    tree = cKDTree(P[good]); _, j = tree.query(P[int(ix)])
    return int(good[int(j)])
