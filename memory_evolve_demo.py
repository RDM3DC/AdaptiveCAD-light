"""Headless exercise for AdaptiveCAD memory evolution workflow.

Loads the installed adaptivecad_core add-on (from Blender's user scripts),
rebuilds the demo scene, runs Cut and memory evolution, then bakes a
heatmap so the accumulated field can be inspected.
"""

import os
import sys
import bpy
from mathutils import Vector


ADDON_NAME = "adaptivecad_core"


def _clear_module(name: str) -> None:
    for mod_name in list(sys.modules.keys()):
        if mod_name == name or mod_name.startswith(f"{name}."):
            sys.modules.pop(mod_name)


def _enable_installed_addon() -> None:
    if ADDON_NAME in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_disable(module=ADDON_NAME)
    _clear_module(ADDON_NAME)
    bpy.ops.preferences.addon_enable(module=ADDON_NAME)
    module = sys.modules.get(ADDON_NAME)
    module_path = getattr(module, "__file__", "<unknown>") if module else "<unloaded>"
    print(f"Enabled {ADDON_NAME} from: {module_path}")


def _base_output_dir() -> str:
    script_dir = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(script_dir, "adaptivecad_core", "adaptivecad_out"))
    os.makedirs(path, exist_ok=True)
    return path


OUT_DIR = _base_output_dir()
BLEND_OUT = os.path.join(OUT_DIR, "adaptivecad_memory.blend")
RENDER_OUT = os.path.join(OUT_DIR, "heatmap_memory.png")


def set_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def fill_and_extrude(obj: bpy.types.Object, height: float = 0.15) -> None:
    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill()
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, height)})
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


def _pick_heatmap_source() -> str:
    heatmap_rna = bpy.ops.adaptivecad.heatmap.get_rna_type()
    source_prop = heatmap_rna.properties.get("source_attr")
    if not source_prop:
        return "curv_memory"
    source_ids = [item.identifier for item in source_prop.enum_items]
    print(f"Heatmap sources available: {source_ids}")
    if "pia_memory" in source_ids:
        return "pia_memory"
    return source_prop.default or "curv_memory"


def run() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _enable_installed_addon()

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    left = bpy.context.active_object
    left.name = "PiA_Left"
    fill_and_extrude(left, height=0.2)

    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    right = bpy.context.active_object
    right.name = "PiA_Right"
    right.location.x = 1.75
    fill_and_extrude(right, height=0.2)

    set_active(left)
    right.select_set(True)
    bpy.ops.adaptivecad.cut()

    set_active(left)
    evolve_op = getattr(bpy.ops.adaptivecad, "evolve_memory", None)
    if evolve_op is None:
        print("Warning: adaptivecad.evolve_memory operator not found; skipping evolution")
    else:
        evolve_rna = bpy.ops.adaptivecad.evolve_memory.get_rna_type()
        prop_names = [p.identifier for p in evolve_rna.properties if p.identifier not in {"rna_type"}]
        print(f"Evolve memory properties: {prop_names}")
        result = bpy.ops.adaptivecad.evolve_memory()
        print(f"adaptivecad.evolve_memory() -> {result}")

    heatmap_source = _pick_heatmap_source()
    print(f"Baking heatmap from attribute: {heatmap_source}")
    bpy.ops.adaptivecad.heatmap(
        source_attr=heatmap_source,
        attr_abs=False if heatmap_source == "pia_memory" else True,
        vcol_name="heatmap_memory",
        mat_name="AdaptiveCAD_Memory_Heatmap"
    )

    ensure_camera_light()
    scene = bpy.context.scene
    engine_items = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
    scene.render.filepath = RENDER_OUT
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
    print(f"Saved blend: {BLEND_OUT}")
    print(f"Saved render: {RENDER_OUT}")


if __name__ == "__main__":
    run()
