# AdaptiveCAD-light – Copilot Guide

## Architecture & Key Modules
- `adaptivecad_playground/app.py` is the PySide6 entry point; it wires the `Canvas` widget as the central view, the `Controls` dock on the right, and defines menus/toolbar plus status widgets.
- `adaptivecad_playground/widgets.py` hosts the interactive canvas, parameter controls, and serialization helpers. Shapes live in `Canvas._shapes` as `Shape` dataclasses; temporary previews use `_temp_shape`. Undo/redo snapshots must serialize both shapes and global canvas state (params, snap, grid).
- `adaptivecad_playground/tools.py` contains per-tool controllers. Tools access the canvas API (`add_circle`, `shape_from_curve`, `set_temp_shape`, `emit_status_from_shape`, etc.). When creating new tools, hook into `Canvas.set_tool` via `_instantiate_tool` and respect snapping with `canvas.snap_point`.
- `adaptivecad_playground/geometry.py` houses all math helpers (π-adaptive kernel, circle/curve sampling, lengths, areas). Reuse these utilities for any metric you expose; do not duplicate math logic inside UI files.

## Interaction Patterns
- All drawing updates must flow through `Canvas` methods so undo/redo snapshots stay valid. Call `Canvas._push_history()` before mutating `_shapes` or other persisted state.
- Status bar metrics come from `_shape_metrics`; extend this when introducing new shape/dimension types so readouts stay in sync.
- Snapping is controlled by `Canvas.snap_point` and the `_snap_to_grid` flag; preserve this behavior for new features (call `snap_point` on any cursor-derived position).
- Export routines live in `Canvas.export_png` / `export_json`; ensure new persistent data (e.g., dimensions, styles) is added to both serialization and `_snapshot`/`_restore` paths.
- UI tooltips and labels use concise, user-facing language. Follow the existing pattern of adding `setToolTip`/`setStatusTip` when introducing actions or controls.

## Workflow & Commands
- Create/activate the virtual environment and install deps:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Launch the playground with `python adaptivecad_playground/app.py`. Use `python -m compileall -q adaptivecad_playground` for a quick syntax check before commits.

## Conventions & Tips
- Use dataclasses for persisted entities (see `Shape`); keep numpy arrays for sampled geometry to leverage vectorized ops.
- Maintain screen-space sizing (e.g., text, arrowheads) by converting logical sizes in the painter rather than storing zoom-dependent values.
- When adding major capabilities (dimensions, additional exports), factor shared logic into dedicated modules (e.g., `dimensions.py`) instead of inflating the canvas class.
- Keep all file I/O UTF-8 and stick to ASCII in new source unless you have a strong reason (existing πₐ terminology is acceptable).
- Document new interaction flows briefly in `README.md` so manual testing steps remain discoverable.

## Validation Before PRs
- Run the app and exercise the new tools (draw, adjust sliders, undo/redo, export JSON/PNG) to confirm no regressions.
- Ensure JSON exports round-trip: save, reload (if/when import exists), and verify schema keys are stable.
- Update GitHub issues/checklists in sync with new functionality as outlined in project prompts (e.g., Dimensions v0.1 scope).

Please review these notes and let me know if any important workflow or convention is missing or unclear so I can refine the guide.
