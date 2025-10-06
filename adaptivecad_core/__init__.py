bl_info = {
    "name": "AdaptiveCAD Core (πₐ) + Heatmap",
    "description": "Curve-native modeling: Add/Cut/Join with πₐ fields, edge attributes, and heatmap visualizer.",
    "author": "RDM3DC",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AdaptiveCAD",
    "category": "Object"
}

import bpy
from .ops import (
    ADAPTIVECAD_OT_add,
    ADAPTIVECAD_OT_cut,
    ADAPTIVECAD_OT_join,
    ADAPTIVECAD_OT_heatmap,
    ADAPTIVECAD_PT_panel,
)

classes = (
    ADAPTIVECAD_OT_add,
    ADAPTIVECAD_OT_cut,
    ADAPTIVECAD_OT_join,
    ADAPTIVECAD_OT_heatmap,
    ADAPTIVECAD_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
