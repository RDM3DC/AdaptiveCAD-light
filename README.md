# AdaptiveCAD Playground (πₐ)

Interactive PySide6 sandbox for experimenting with curvature-adaptive geometry. Draw πₐ-driven circles and Bezier-like curves, tweak parameters live, and export the scene.

## Features
- πₐ Circle and πₐ Curve tools with live previews.
- Parameter sliders for α, μ, and k₀.
- Dimension annotations can switch between inches and millimeters via the Parameters dock.
- Tiny Sketch Mode selector (XY / YZ / ZX) prepares profiles for future revolve tools.
- Adaptive curvature and material memory field sampling; hover with Measure to inspect `kappa`, `pi_a`, and memory values.
- Snap-to-grid toggle (on by default) with grid overlay.
- Zoom controls: `Ctrl` + mouse wheel, `Ctrl` + `+` / `-`, and `Ctrl+0` to reset.
- Undo/redo shortcuts (`Ctrl+Z` / `Ctrl+Y`).
- PNG and JSON export (JSON captures shapes, params, snap settings).

## Getting Started
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python adaptivecad_playground/app.py
```

### Optional: Enable GPU Acceleration
- Install a CuPy wheel that matches your local CUDA runtime, e.g. `pip install cupy-cuda12x`.
- (Recommended) Set `CUDA_VISIBLE_DEVICES=0,1` before launching to pin the two 3090 Ti boards.
- To temporarily force CPU mode, export `ADAPTIVECAD_DISABLE_GPU=1` prior to running the playground.

## Repository Layout
```
adaptivecad_playground/
  app.py          # Main window and menus
  widgets.py      # Canvas, controls, grid, history
  tools.py        # Interactive tools (select/circle/curve)
  adaptive_fields.py # Curvature/material field synthesis for πₐ metrics
  geometry.py     # πₐ math helpers and sampling routines
adaptivecad_core/
  __init__.py
  headless_demo.py
  ops.py
  revolve.py        # Adaptive revolve helper (θ sampling, seam logic)
  fields.py         # Curvature + memory pack for headless tools
  metrics.py        # Adaptive arc length helpers
  prototype_cut_join.py # Placeholder cut/add/join routines
  demo_sketch_to_revolve.py # Pure Python revolve OBJ demo
assets/
  README.txt      # Placeholder for icons
```

## Export Formats
- **PNG**: full canvas snapshot.
- **JSON**: includes canvas size, parameter values, snap settings, and sampled points for each shape.

## License
MIT (update as needed).
