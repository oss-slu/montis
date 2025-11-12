from __future__ import annotations
import numpy as np
from typing import Union

def risk_from_slope(s: Union[np.ndarray, float]) -> np.ndarray:
    s = np.clip(s, 0, 90)
    lo = 1/(1+np.exp(-(s-22)/3))
    hi = 1/(1+np.exp(-(40-s)/3))
    return np.clip(lo*hi, 0, 1)
