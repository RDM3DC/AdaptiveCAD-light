"""Headless AdaptiveCAD demo for Blender CLI runs.

Steps executed:
1) Enable the adaptivecad_core add-on.
2) Add two PiA primitives.
3) Fill and extrude each so the boolean has volume.
4) Run the Adaptive Cut operator (records curv_memory edge attribute).
5) Bake a heatmap material from curv_memory.
6) Save the .blend and render a still.
"""

import bpy
from mathutils import Vector
import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _base_output_dir() -> str:
    script_dir = os.path.dirname(bpy.data.filepath or __file__)
    path = os.path.abspath(os.path.join(script_dir, "adaptivecad_out"))
    os.makedirs(path, exist_ok=True)
    return path

OUT_DIR = _base_output_dir()
BLEND_OUT = os.path.join(OUT_DIR, "adaptivecad_demo.blend")
RENDER_OUT = os.path.join(OUT_DIR, "heatmap.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def fill_and_extrude(obj: bpy.types.Object, height: float = 0.1) -> None:
    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill()
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0.0, 0.0, height)}
    )
    bpy.ops.object.mode_set(mode='OBJECT')

def ensure_camera_light() -> None:
    scene = bpy.context.scene
    if "AdaptiveCAD_Camera" not in bpy.data.objects:
        cam_data = bpy.data.cameras.new("AdaptiveCAD_Camera")
        cam = bpy.data.objects.new("AdaptiveCAD_Camera", cam_data)
        bpy.context.collection.objects.link(cam)
        cam.location = (0, -6, 2)
        cam.rotation_euler = (1.1, 0.0, 0.0)
        scene.camera = cam
    if "AdaptiveCAD_Light" not in bpy.data.objects:
        light_data = bpy.data.lights.new("AdaptiveCAD_Light", type='AREA')
        light = bpy.data.objects.new("AdaptiveCAD_Light", light_data)
        bpy.context.collection.objects.link(light)
        light.location = (3, -2, 5)
        light.data.energy = 1000


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)

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

    try:
        adaptive_module = importlib.import_module(addon_module_name)
        importlib.reload(adaptive_module)
        if hasattr(adaptive_module, "register"):
            adaptive_module.register()
        else:
            raise RuntimeError("adaptivecad_core missing register()")
        print(f"Registered adaptivecad_core from: {adaptive_module.__file__}")
    except Exception as exc:
        print("Could not register adaptivecad_core from local path:", exc)
        try:
            bpy.ops.preferences.addon_enable(module=addon_module_name)
        except Exception as fallback_exc:
            print("Fallback enable failed:", fallback_exc)
            sys.exit(1)

    # Create PiA primitives
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    left = bpy.context.active_object
    left.name = "PiA_Left"

    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    right = bpy.context.active_object
    right.name = "PiA_Right"
    right.location.x = 1.8

    fill_and_extrude(left, height=0.15)
    fill_and_extrude(right, height=0.15)

    # Run Cut (PiA)
    set_active(left)
    right.select_set(True)
    bpy.ops.adaptivecad.cut()

    # Bake heatmap
    set_active(left)
    bpy.ops.adaptivecad.heatmap(
        source_attr='curv_memory',
        attr_abs=True,
        vcol_name="heatmap",
        mat_name="AdaptiveCAD_Heatmap_Mat",
    )

    ensure_camera_light()
    scene = bpy.context.scene
    engine_items = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
    }
    engine_name = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
    scene.render.engine = engine_name
    scene.render.filepath = RENDER_OUT
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)

    print("Saved:", BLEND_OUT)
    print("Rendered:", RENDER_OUT)


if __name__ == "__main__":
    run()
