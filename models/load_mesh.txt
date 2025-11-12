from __future__ import annotations
import numpy as np
import trimesh
from .as_mesh import _as_mesh
from .normals import estimate_normals

def load_mesh_to_arrays(path: str):
    """
    Load .ply or .obj (and friends) into arrays.
    Returns (P[N,3], idx[3T] or None, normals[N,3], bounds dict)
    """
    m = trimesh.load(path, process=False)
    m = _as_mesh(m)

    if isinstance(m, trimesh.points.PointCloud):
        P = np.asarray(m.vertices, dtype=np.float32)
        idx = None
        normals = estimate_normals(P)
    else:
        if not hasattr(m, "vertices"):
            m = trimesh.load_mesh(path, process=False)
        P = np.asarray(m.vertices, dtype=np.float32)
        idx = None
        if hasattr(m, "faces") and m.faces is not None and len(m.faces) > 0:
            idx = np.asarray(m.faces, dtype=np.uint32).reshape(-1)
        normals = None
        if hasattr(m, "vertex_normals") and m.vertex_normals is not None and len(m.vertex_normals) == len(P):
            normals = np.asarray(m.vertex_normals, dtype=np.float32)
        if normals is None:
            if idx is not None:
                _ = m.vertex_normals  # triggers compute
                normals = np.asarray(m.vertex_normals, dtype=np.float32)
            else:
                normals = estimate_normals(P)

    bounds = dict(
        min=dict(x=float(P[:,0].min()), y=float(P[:,1].min()), z=float(P[:,2].min())),
        max=dict(x=float(P[:,0].max()), y=float(P[:,1].max()), z=float(P[:,2].max())),
    )
    return P, idx, normals, bounds
