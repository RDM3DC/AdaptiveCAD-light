from adaptivecad import adaptive_shape, slider, dropdown
import numpy as np

@adaptive_shape("Geometric Unity Möbius")
def mobius_unity_shape(radius=40.0, half_width=8.0, twists=1.5,
                       gamma=0.25, tau=0.5, proj_mode="hybrid",
                       samples_major=720, samples_width=64):
    """Interactive Möbius shape with 4D projection blending."""

    u = np.linspace(0, 2*np.pi, samples_major, endpoint=True)
    v = np.linspace(-half_width, half_width, samples_width)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.T, vv.T

    t4 = gamma * np.sin(uu * twists / 2)

    # --- base Euclidean coordinates
    x = (radius + vv * np.cos(twists * uu / 2)) * np.cos(uu)
    y = (radius + vv * np.cos(twists * uu / 2)) * np.sin(uu)
    z = vv * np.sin(twists * uu / 2)

    # --- projection morphing
    phase = np.exp(1j * uu * twists)
    if proj_mode == "euclidean":
        x3, y3, z3 = x, y, z
    elif proj_mode == "lorentz":
        x3 = x * np.cosh(t4) - z * np.sinh(t4)
        y3 = y
        z3 = z * np.cosh(t4) - x * np.sinh(t4)
    elif proj_mode == "complex":
        x3 = np.real(x + phase * 0.4)
        y3 = np.imag(y + phase * 0.4)
        z3 = z + np.real(phase) * 0.2
    else:  # hybrid
        xE, yE, zE = x, y, z
        xL = x * np.cosh(t4) - z * np.sinh(t4)
        yL = y
        zL = z * np.cosh(t4) - x * np.sinh(t4)
        xC = np.real(x + phase * 0.4)
        yC = np.imag(y + phase * 0.4)
        zC = z + np.real(phase) * 0.2
        x3 = (1 - tau) * xE + tau * 0.5 * (xL + xC)
        y3 = (1 - tau) * yE + tau * 0.5 * (yL + yC)
        z3 = (1 - tau) * zE + tau * 0.5 * (zL + zC)

    vertices = np.column_stack([x3.ravel(), y3.ravel(), z3.ravel()])
    faces = []
    for i in range(samples_major - 1):
        for j in range(samples_width - 1):
            a = i * samples_width + j
            b = a + 1
            c = a + samples_width
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))

    return vertices, faces

# --- GUI Controls ---
slider("tau", 0.0, 1.0, 0.01, label="4D Projection Morph")
dropdown("proj_mode", ["euclidean", "lorentz", "complex", "hybrid"], label="Projection Mode")
