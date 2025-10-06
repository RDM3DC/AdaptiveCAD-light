bl_info = {
    "name": "AdaptiveCAD Core (πₐ) + Heatmap",
    "description": "Curve-native modeling: Add/Cut/Join with πₐ fields, edge attributes, and heatmap visualizer.",
    "author": "RDM3DC",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "category": "Object"
}

import bpy
from .ops import (
    ADAPTIVECAD_PiAProps,
    ADAPTIVECAD_OT_add,
    ADAPTIVECAD_OT_cut,
    ADAPTIVECAD_OT_join,
    ADAPTIVECAD_OT_heatmap,
    ADAPTIVECAD_OT_evolve_memory,
    ADAPTIVECAD_PT_panel,
)

classes = (
    ADAPTIVECAD_OT_add,
    ADAPTIVECAD_OT_cut,
    ADAPTIVECAD_OT_join,
    ADAPTIVECAD_OT_heatmap,
    ADAPTIVECAD_OT_evolve_memory,
    ADAPTIVECAD_PT_panel,
)

def register():
    bpy.utils.register_class(ADAPTIVECAD_PiAProps)
    bpy.types.Scene.pia = bpy.props.PointerProperty(type=ADAPTIVECAD_PiAProps)
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    if hasattr(bpy.types.Scene, "pia"):
        del bpy.types.Scene.pia
    bpy.utils.unregister_class(ADAPTIVECAD_PiAProps)

if __name__ == "__main__":
    register()
