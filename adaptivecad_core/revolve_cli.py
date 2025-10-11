"""Headless revolve demo using the AdaptiveCAD revolve operator.

Run with:
    blender --background --factory-startup --python revolve_cli.py

Outputs a revolved mesh to adaptivecad_out/revolve_demo.blend.
"""
from __future__ import annotations

import importlib
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import bpy

from adaptivecad_playground.sketch_mode import canonical_sketch_profiles, lift_polyline, SketchPlane


def _output_path(filename: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "adaptivecad_out")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)


def _register_local_addon() -> None:
    addon_module_name = "adaptivecad_core"
    addon_dir = os.path.abspath(os.path.dirname(__file__))
    addon_parent = os.path.dirname(addon_dir)
    if addon_parent not in sys.path:
        sys.path.insert(0, addon_parent)

    if addon_module_name in bpy.context.preferences.addons:
        try:
            bpy.ops.preferences.addon_disable(module=addon_module_name)
        except Exception:
            pass

    for mod_name in list(sys.modules.keys()):
        if mod_name == addon_module_name or mod_name.startswith(f"{addon_module_name}."):
            sys.modules.pop(mod_name)

    module = importlib.import_module(addon_module_name)
    importlib.reload(module)
    if hasattr(module, "register"):
        module.register()
    else:
        raise RuntimeError("adaptivecad_core missing register()")


def _make_profile() -> bpy.types.Object:
    profiles = canonical_sketch_profiles()
    spec = profiles["vase_profile"]
    plane = spec["plane"]
    if isinstance(plane, str):
        plane = SketchPlane[plane]
    points_2d = spec["points"]
    pts = lift_polyline(points_2d, plane)
    verts = [tuple(map(float, pt)) for pt in pts]
    edges = [(i, i + 1) for i in range(len(verts) - 1)]
    mesh = bpy.data.meshes.new("RevolveProfile")
    mesh.from_pydata(verts, edges, [])
    mesh.update(calc_edges=True, calc_edges_loose=True)

    obj = bpy.data.objects.new("RevolveProfile", mesh)
    bpy.context.collection.objects.link(obj)

    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def run() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _register_local_addon()

    profile = _make_profile()

    result = bpy.ops.adaptivecad.revolve(
        axis_mode='Z',
        angle=320.0,
        theta_offset=0.0,
        tolerance=0.0005,
        eta_k=0.45,
        eta_m=0.35,
        smart_seam=True,
        cap_start=True,
        cap_end=True,
        keep_profile=False,
    )
    if result != {'FINISHED'}:
        raise RuntimeError(f"Revolve operator failed: {result}")

    out_path = _output_path("revolve_demo.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("Saved revolve demo to", out_path)


if __name__ == "__main__":
    run()
