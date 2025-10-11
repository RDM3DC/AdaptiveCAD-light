# CurvedUI v0.1 — “No Straight Lines” Adaptive UI

This pack adds an **organic, curved** UI layer to AdaptiveCAD-light:
- Superellipse (“squircle”) panels and borders
- πₐ‑warped **arc ribbons** for tool beads
- Adaptive curvature (more curved when idle, tighter corners when busy)
- Paints with **paper-space strokes** so it looks crisp at any zoom

> This is a UX layer; it doesn’t alter geometry or math claims. For ARP/NSE, keep language consistent with the ARP lens (numerical stabilization, not a Clay proof).

## Files
```
adaptivecad_playground/
  curved_ui.py         # geometry helpers: superellipse, arc layout, curvature scheduler, πₐ warp
  curved_widgets.py    # CurvedViewportOverlay, CurvedPanel, CurvedButton
  curved_theme.py      # theme presets
  icons/
    arc_ribbon.svg
    orb_button.svg
    panel_squircle.svg
```

## Quick wiring (5 minutes)

1) **Drop files** into `adaptivecad_playground/`.
2) In your Canvas (`widgets.py`), add an overlay child that sits on top of your drawing area:

```python
from adaptivecad_playground.curved_widgets import CurvedViewportOverlay

class Canvas(QWidget):
    def __init__(self, ...):
        ...
        self.overlay = CurvedViewportOverlay(self, get_params=self.params)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.overlay.setGeometry(self.rect())
```

3) **Adaptivity** (optional): pump activity into the overlay when the user is active:

```python
def mouseMoveEvent(self, ev):
    super().mouseMoveEvent(ev)
    # crude activity metric based on mouse speed
    if hasattr(self, "overlay"):
        self.overlay.adapt(input_energy=0.2)
```

4) **Use CurvedPanel/CurvedButton** for your side docks or popups:

```python
from adaptivecad_playground.curved_widgets import CurvedPanel, CurvedButton
panel = CurvedPanel(parent=self)
panel.setGeometry(20, 60, 260, 140); panel.show()
```

## Notes
- All outlines are curved superellipses; even buttons are capsules or orbs. No straight lines.
- The ribbon bead positions use πₐ‑warped angles (`curved_ui.arc_positions`), so your UI *visually* reflects adaptive π.
- CurvatureState gives you a simple way to “breathe” the UI based on interaction.

## Roadmap
- v0.2: metaball “blob dock,” smooth boolean unions for panels, curved sliders, animated help chips
- v0.3: curvature driven by tool context (e.g., stronger curvature during edit ops), sound-reactive pulses
