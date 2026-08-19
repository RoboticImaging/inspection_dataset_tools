"""Alignment primitives used to put a prediction in the gt gauge.

`Sim3`/`umeyama`/`fit_cameras` are pure geometry with no notion of "evaluation" -- they live in
`inspection_viz.align` (alongside `Camera`) so `inspection_viz.Viewer` can use the exact same fit
to line a source up with gt on screen, and are re-exported here since every metric already does
`from .align import Sim3, umeyama`.

Relative pose metrics (RRA/RTA) are invariant to the gauge, so `rotation_angles` -- used only by
those -- stays here rather than moving down with the rest.
"""

import numpy as np

from inspection_viz.align import Sim3, fit_cameras, umeyama

__all__ = ["Sim3", "fit_cameras", "rotation_angles", "umeyama"]


def rotation_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between two stacks of (...,3,3) rotations."""
    tr = np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
