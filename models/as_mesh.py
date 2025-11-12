from __future__ import annotations
import trimesh
def _as_mesh(scene_or_mesh):
    """Convert a trimesh.Scene to a single Trimesh by concatenation."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            raise ValueError("scene has no geometry")
        return trimesh.util.concatenate(geoms)
    return scene_or_mesh