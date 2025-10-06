"""Headless Join demo: create two PiA discs, join them, and bake weld heatmap."""

import os
import sys
import importlib
from math import radians

import bpy


def set_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def shade_smooth(obj: bpy.types.Object, angle_degrees: float = 35.0) -> None:
    set_active(obj)
    bpy.ops.object.shade_smooth()
    mesh = obj.data
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = radians(angle_degrees)


def solidify_circle(obj: bpy.types.Object, height: float = 0.15) -> None:
    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill()
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, height)})
    bpy.ops.object.mode_set(mode='OBJECT')


def ensure_camera_light() -> None:
    scene = bpy.context.scene
    if scene.camera is None:
        cam_data = bpy.data.cameras.new("AdaptiveCAD_JoinCam")
        cam = bpy.data.objects.new("AdaptiveCAD_JoinCam", cam_data)
        bpy.context.collection.objects.link(cam)
        scene.camera = cam
        cam.location = (0.0, -6.0, 2.0)
        cam.rotation_euler = (1.1, 0.0, 0.0)
    if "AdaptiveCAD_JoinKey" not in bpy.data.objects:
        light_data = bpy.data.lights.new("AdaptiveCAD_JoinKey", type='AREA')
        light = bpy.data.objects.new("AdaptiveCAD_JoinKey", light_data)
        bpy.context.collection.objects.link(light)
        light.location = (3.0, -2.0, 5.0)
        light.data.energy = 1000.0


def enable_addon() -> None:
    addon_name = "adaptivecad_core"
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Ensure we always load the workspace copy rather than a globally installed add-on.
    if addon_name in sys.modules:
        module = importlib.reload(sys.modules[addon_name])
    else:
        module = importlib.import_module(addon_name)

    if addon_name in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_disable(module=addon_name)

    if hasattr(module, "register"):
        module.register()

    print(f"Enabled {addon_name} from {module.__file__}")


def run() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_addon()

    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    obj_a = bpy.context.active_object
    obj_a.name = "Join_A"
    solidify_circle(obj_a)

    bpy.ops.adaptivecad.add(radius=1.2, segments=128)
    obj_b = bpy.context.active_object
    obj_b.name = "Join_B"
    obj_b.location.x = 1.6
    solidify_circle(obj_b)

    set_active(obj_a)
    obj_b.select_set(True)
    bpy.ops.adaptivecad.join()

    joined = bpy.context.active_object
    shade_smooth(joined)
    bpy.ops.adaptivecad.heatmap(
        source_attr='weld_score',
        attr_abs=True,
        vcol_name="heatmap",
        mat_name="AdaptiveCAD_Heatmap_Mat",
    )

    ensure_camera_light()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "adaptivecad_core", "adaptivecad_out"))
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    engine_items = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
    }
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
    scene.render.filepath = os.path.join(out_dir, "join_weld_heatmap.png")
    bpy.ops.render.render(write_still=True)

    blend_path = os.path.join(out_dir, "join_demo.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print("Saved join demo to", out_dir)
    print("Rendered heatmap:", scene.render.filepath)


if __name__ == "__main__":
    run()
