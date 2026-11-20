"""MuJoCo bodies of registry objects (assets/registry.jsonl): textured visual mesh + collision geoms.

scene._add_object must delegate here for ObjSpecs with attrs["asset"] (soarm.objects.asset_spec):
    if o.attrs.get("asset"): return asset_mj.add_object(spec, o, friction)
(requested from soarm-core; until that lands on soarm-div, tools call install_scene_hook()).
Geoms: `obj:<name>:0..` collision (group 3, hidden; carries the mass), `obj:<name>:vis` visual mesh (group 2, no
contacts), so segmentation and the held-object check map both to the task object.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .objects import ObjSpec, asset_root, registry_row

VISUAL_GROUP, COLLISION_GROUP = 2, 3
# matte plastic / wood: low specular, broad highlight
SHADING = {"plastic": dict(specular=0.08, shininess=0.2, reflectance=0.0),
           "wood": dict(specular=0.04, shininess=0.1, reflectance=0.0)}


def add_object(spec: mujoco.MjSpec, o: ObjSpec, table_friction: float) -> mujoco.MjsBody:
    """Free body `o.name` at the origin (the caller sets pos/quat). Friction = attrs["friction"] (MuJoCo takes the max of
    the two geoms in a contact, so the table/pad friction is a floor), else the scene's table friction."""
    row = registry_row(o.attrs["asset"])
    root = asset_root()
    mesh_file = root / row["visual"]["mesh"]
    if not mesh_file.is_file():
        raise FileNotFoundError(f"{mesh_file}: run `python -m assets.build_cubes` / `assets.import_objects` "
                                f"(SOARM_ASSETS={root})")
    k = float(o.attrs["scale"]) if row["shape"] == "mesh" else float(o.size[0]) / float(row["dims_m"][0])
    mat = spec.add_material(name=f"mat:{o.name}")
    for key, val in SHADING.get(row.get("material"), SHADING["plastic"]).items():
        setattr(mat, key, val)
    if row["visual"].get("texture"):
        tex = spec.add_texture(name=f"tex:{o.name}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               file=str(root / row["visual"]["texture"]))
        tex.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR    # texels as-is, like the primitives' rgba
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    else:
        mat.rgba = [0.6, 0.6, 0.6, 1.0]
    mesh = spec.add_mesh(name=f"mesh:{o.name}", file=str(mesh_file), scale=[k, k, k])
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    body = spec.worldbody.add_body(name=o.name)
    body.add_freejoint(name=f"{o.name}:free")
    friction = float(o.attrs.get("friction", table_friction))
    cols = row["collision"]
    geoms = []
    if [c.get("type") if isinstance(c, dict) else "mesh" for c in cols] == ["box"]:
        geoms.append((body.add_geom(name=f"obj:{o.name}:0", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=(0.5 * np.asarray(o.size)).tolist()), 1.0))
    elif all(isinstance(c, str) for c in cols):        # convex hulls (CoACD), mass split by hull volume
        vols = np.asarray(row.get("collision_volumes_m3") or [1.0] * len(cols), float)
        for i, (path, w) in enumerate(zip(cols, vols / vols.sum())):
            hull = spec.add_mesh(name=f"hull:{o.name}:{i}", file=str(root / path), scale=[k, k, k])
            geoms.append((body.add_geom(name=f"obj:{o.name}:{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                                        meshname=hull.name), float(w)))
    else:
        raise NotImplementedError(f"{row['id']}: collision {cols!r}")
    for g, w in geoms:
        g.group = COLLISION_GROUP
        g.rgba = [0.5, 0.5, 0.5, 1.0]
        g.friction = [friction, 0.005, 0.0001]
        g.condim = 3
        g.solref = [0.01, 1.0]
        g.mass = o.mass * w
    v = body.add_geom(name=f"obj:{o.name}:vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                      material=mat.name, group=VISUAL_GROUP)
    v.contype = v.conaffinity = 0
    v.mass = 0.0
    v.density = 0.0
    return body


def install_scene_hook() -> None:
    """Route registry objects of scene.build_scene through add_object (idempotent; no-op once scene.py delegates)."""
    from . import scene
    orig = scene._add_object
    if getattr(orig, "_asset_hook", False):
        return

    def _add_object(spec, o, friction):
        if o.attrs.get("asset"):
            add_object(spec, o, friction)
        else:
            orig(spec, o, friction)

    _add_object._asset_hook = True
    scene._add_object = _add_object
