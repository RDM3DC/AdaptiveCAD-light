# Placeholder# EditOps v0.1 Hookup Guide

This patch introduces the first set of drafting edit tools for AdaptiveCAD-light:

- **Join** – merge segments/polylines whose endpoints are within tolerance.
- **Break** – split a segment/polyline at a picked point.
- **Trim** – shorten a segment back to the nearest cutter intersection.
- **Extend** – lengthen a segment until it meets a boundary curve.
- **Boolean helpers** – union/difference/intersection for polygons (via Shapely when available, convex clip fallback for intersections).

## Files

- `adaptivecad_playground/intersect2d.py` – low-level intersection and projection helpers.
- `adaptivecad_playground/edit_ops.py` – core Euclidean edit operations.
- `adaptivecad_playground/boolean2d.py` – polygon Boolean utilities.
- `adaptivecad_playground/edit_tools.py` – UI scaffolding that wires edit ops to the canvas selection & history system.
- `adaptivecad_playground/icons/join.svg`, `break.svg`, `trim.svg`, `extend.svg` – toolbar glyphs.

## Canvas Hooks (summary)

`Canvas` now exposes a small API for edit tools:

- `get_selection()` / `set_selection(payloads)` – round-trip friendly dictionaries (type, id, points, etc.).
- `update_scene(add_list, remove_list)` – atomic edits with undo/redo history and selection updates.
- `world_from_event(event)` – snapped world coordinate for pointer clicks.
- `osnap_targets()` – shared target generator (end/mid/center) for object snap.
- `set_pointer_tool(tool)` – installs a temporary mouse handler (Break/Trim/Extend).

Selections support multi-pick (Shift-click to add/remove). Edit ops record undo snapshots and reseed IDs for new shapes.

## Toolbar Wiring

`Main._make_toolbar` adds Join/Break/Trim/Extend buttons next to existing drawing tools. When Break/Trim/Extend are activated, the canvas switches to Select mode, installs the respective pointer tool, and StatusBar cues guide you through the stages (target → cutter/boundary).

## Boolean Ops

`boolean2d.py` checks for Shapely at runtime. If present, `poly_union`, `poly_difference`, and `poly_intersection` return lists of exterior rings. Without Shapely, `poly_intersection` falls back to Sutherland–Hodgman for convex clips; `union`/`difference` raise `NotImplementedError` (planned for v0.2).

## Testing Checklist

1. Draw a chain of segments/polylines → select → Join → expect a single polyline.
2. Break a segment in the middle → expect two segments.
3. Trim: select target segment (Shift for multiple), then cutter (segment/polyline/circle) → trimmed to intersection.
4. Extend: select target, then boundary → extended to meet boundary.
5. (Optional) If Shapely installed, run polygon Add/Cut/Intersect using `boolean2d` helpers.
