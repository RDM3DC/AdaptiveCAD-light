"""Pure-Python demo: build an OBJ via the adaptive revolve helper."""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _load_revolve_adaptive():
    module_name = "adaptivecad_core.revolve"
    file_path = os.path.join(BASE_DIR, "adaptivecad_core", "revolve.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to locate adaptivecad_core.revolve")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.revolve_adaptive


revolve_adaptive = _load_revolve_adaptive()

from adaptivecad_playground.sketch_mode import SketchPlane, canonical_sketch_profiles, lift_polyline


def _canonical_profile() -> np.ndarray:
    profiles = canonical_sketch_profiles()
    spec = profiles["vase_profile"]
    plane = spec["plane"]
    if isinstance(plane, str):
        plane = SketchPlane[plane]
    points_2d = spec["points"]
    return lift_polyline(points_2d, plane)


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as handle:
        for v in vertices:
            handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
            handle.write(f"f {a} {b} {c}\n")


def main() -> None:
    profile = _canonical_profile()
    result = revolve_adaptive(
        profile_pts=profile,
        axis_point=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 0.0, 1.0),
        angle=2.0 * np.pi,
        tol=1e-3,
        smart_seam=True,
        cap_start=True,
        cap_end=True,
    )
    obj_path = "adaptive_revolve.obj"
    _write_obj(obj_path, result.vertices, result.faces)
    print(
        "Wrote",
        obj_path,
        "| verts:",
        len(result.vertices),
        "tris:",
        len(result.faces),
        "segments:",
        result.segments,
    )


if __name__ == "__main__":
    main()
