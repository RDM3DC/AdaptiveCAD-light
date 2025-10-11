bl_info = {
    "name": "AdaptiveCAD Core (πₐ) + Heatmap",
    "description": "Curve-native modeling: Add/Cut/Join with πₐ fields, edge attributes, and heatmap visualizer.",
    "author": "RDM3DC",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AdaptiveCAD",
    "category": "Object"
}

try:  # pragma: no cover - Blender-only dependency
    import bpy  # type: ignore
except ModuleNotFoundError:  # Running outside Blender (e.g., CLI usage)
    bpy = None  # type: ignore

if bpy is not None:
    from .ops import (  # noqa: WPS433 - re-export for Blender add-on
        ADAPTIVECAD_OT_add,
        ADAPTIVECAD_OT_cut,
        ADAPTIVECAD_OT_join,
        ADAPTIVECAD_OT_revolve,
        ADAPTIVECAD_OT_heatmap,
        ADAPTIVECAD_PT_panel,
    )

    classes = (
        ADAPTIVECAD_OT_add,
        ADAPTIVECAD_OT_cut,
        ADAPTIVECAD_OT_join,
        ADAPTIVECAD_OT_revolve,
        ADAPTIVECAD_OT_heatmap,
        ADAPTIVECAD_PT_panel,
    )

    def register() -> None:
        for cls in classes:
            bpy.utils.register_class(cls)

    def unregister() -> None:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
else:

    def register() -> None:  # pragma: no cover - guard for non-Blender usage
        raise RuntimeError("adaptivecad_core add-on can only be registered inside Blender")

    def unregister() -> None:  # pragma: no cover - guard for non-Blender usage
        raise RuntimeError("adaptivecad_core add-on can only be unregistered inside Blender")

if __name__ == "__main__":
    register()
