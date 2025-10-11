# AdaptiveCAD – Revolve Modules (π-aware sampling)

This note captures the standalone revolve bundle that shipped the lightweight
FieldPack, adaptive revolve helper, and Blender add-on. The code is now wired
into the repository:

- `adaptivecad_core/revolve.py` — adaptive revolve helper with curvature and
  seam memory support.
- `adaptivecad_core/fields.py` — portable curvature + memory `FieldPack` used by
  headless demos.
- `adaptivecad_core/metrics.py` — `arc_len_adaptive` metric for π-aware lengths.
- `adaptivecad_core/prototype_cut_join.py` — placeholder adaptive cut / add /
  join routines for experimentation.
- `adaptivecad_core/demo_sketch_to_revolve.py` — pure Python OBJ writer using the
  canonical vase sketch profile.
- `adaptive_revolve_addon.py` — minimal Blender operator exposed in the N-panel.

The original quick-start still applies:

```powershell
# Pure Python OBJ demo (writes adaptive_revolve.obj in the working dir)
python adaptivecad_core/demo_sketch_to_revolve.py

# Blender headless revolve demo (outputs adaptivecad_out/revolve_demo.blend)
"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" \
    --background --factory-startup \
    --python adaptivecad_core/revolve_cli.py
```

Key ideas:

- θ segmentation tightens by curvature (`eta_k`) and optional material memory
  feedback (`eta_m`).
- Seam memory can be persisted by passing a `FieldPack` instance to
  `revolve_adaptive`.
- Caps are generated when the profile touches the axis at either end.
- The Blender add-on stamps a `POINT` attribute named `AdaptiveMemory` to mark
  seam vertices; the Python demo simply writes indices for inspection.

The former `requirements.txt` only declared `numpy>=1.20`, which is already
covered by the project dependencies.
