from __future__ import annotations
from typing import List
import numpy as np
from .risk import risk_from_slope

def dijkstra_route(
    G: List[List[int]],
    P: np.ndarray,
    S: np.ndarray,
    start: int,
    goal: int,
    mode: str,
    maxSlope: float,
    riskPow: float,
    soft: bool = True,
):
    """
    FAST  => pure geometric shortest path (no slope gating / penalties)
    SAFE  => geometric distance * (1 + risk penalty), with 'soft' slope penalty above maxSlope
    """
    import heapq

    N = len(G)
    if N == 0:
        return None

    start = int(np.clip(start, 0, N - 1))
    goal  = int(np.clip(goal,  0, N - 1))

    dist = np.full(N, np.inf)
    prev = np.full(N, -1, dtype=np.int32)
    seen = np.zeros(N, bool)
    pq: list[tuple[float, int]] = []

    def edge_cost(u: int, v: int) -> float:
        x1, y1, z1 = P[u]; x2, y2, z2 = P[v]
        dx, dy, dz = (x2-x1), (y2-y1), (z2-z1)
        d3 = float(np.sqrt(dx*dx + dy*dy + dz*dz))

        if mode == "fast":
            return d3

        s1, s2 = float(S[u]), float(S[v])
        if not soft and (s1 > maxSlope or s2 > maxSlope):
            return np.inf

        r = float(risk_from_slope(np.array([s1, s2])).mean()) ** max(riskPow, 1e-6)

        over = max(0.0, max(s1, s2) - maxSlope)
        pen = 1.0
        if soft and over > 0:
            frac = over / max(1.0, 90.0 - maxSlope)
            pen *= (1.0 + 9.0 * (frac ** 2))  # up to 10x

        return d3 * (1.0 + 4.0 * r) * pen

    dist[start] = 0.0
    heapq.heappush(pq, (0.0, start))

    while pq:
        du, u = heapq.heappop(pq)
        if seen[u]:
            continue
        seen[u] = True
        if u == goal:
            break
        for v in G[u]:
            c = edge_cost(u, v)
            if not np.isfinite(c):
                continue
            nd = du + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal]):
        return None

    path: list[int] = []
    v = goal
    while v != -1:
        path.append(int(v))
        v = int(prev[v])
    path.reverse()

    length = 0.0
    gain = 0.0
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        dx, dy, dz = (P[b] - P[a]).tolist()
        length += float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if dz > 0:
            gain += dz

    return dict(path=path, lengthKm=length / 1000.0, gainM=gain)
