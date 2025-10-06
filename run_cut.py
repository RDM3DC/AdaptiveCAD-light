"""Headless Cut demo: subtract a cutter from a target and bake curvature memory heatmap."""

import os
import sys

import bpy

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_join import enable_addon, set_active, solidify_circle, ensure_camera_light, shade_smooth


def make_cut_primitives():
    bpy.ops.adaptivecad.add(radius=1.4, segments=160)
    target = bpy.context.active_object
    target.name = "Cut_Target"
    solidify_circle(target, height=0.35)

    bpy.ops.adaptivecad.add(radius=0.9, segments=128)
    cutter = bpy.context.active_object
    cutter.name = "Cut_Cutter"
    cutter.location = (0.55, 0.0, 0.12)
    cutter.rotation_euler.x = 0.35
    solidify_circle(cutter, height=0.28)
    return target, cutter


def run() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_addon()

    target, cutter = make_cut_primitives()

    set_active(target)
    cutter.select_set(True)
    result = bpy.ops.adaptivecad.cut()
    if {'FINISHED'} != set(result):
        raise RuntimeError("AdaptiveCAD cut operator failed")

    # Remove cutter object to keep the scene tidy.
    bpy.data.objects.remove(cutter, do_unlink=True)

    cut_obj = bpy.context.active_object
    shade_smooth(cut_obj)

    bpy.ops.adaptivecad.heatmap(
        source_attr='curv_memory',
        attr_abs=False,
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
    scene.render.filepath = os.path.join(out_dir, "cut_heatmap.png")
    bpy.ops.render.render(write_still=True)

    blend_path = os.path.join(out_dir, "cut_demo.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print("Saved cut demo to", out_dir)
    print("Rendered heatmap:", scene.render.filepath)


if __name__ == "__main__":
    run()
