# AdaptiveCAD Playground (πₐ)

Interactive PySide6 sandbox for experimenting with curvature-adaptive geometry. Draw πₐ-driven circles and Bezier-like curves, tweak parameters live, and export the scene.

## Features
- πₐ Circle and πₐ Curve tools with live previews.
- Parameter sliders for α, μ, and k₀.
- Snap-to-grid toggle (on by default) with grid overlay.
- Undo/redo shortcuts (`Ctrl+Z` / `Ctrl+Y`).
- PNG and JSON export (JSON captures shapes, params, snap settings).

## Getting Started
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python adaptivecad_playground/app.py
```

## Repository Layout
```
adaptivecad_playground/
  app.py          # Main window and menus
  widgets.py      # Canvas, controls, grid, history
  tools.py        # Interactive tools (select/circle/curve)
  geometry.py     # πₐ math helpers and sampling routines
assets/
  README.txt      # Placeholder for icons
```

## Export Formats
- **PNG**: full canvas snapshot.
- **JSON**: includes canvas size, parameter values, snap settings, and sampled points for each shape.

## License
MIT (update as needed).
