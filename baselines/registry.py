"""Every baseline the pipeline knows about, by name.

Registering a baseline is one line here: the name it predicts under (`predictions/<name>/`, and
the viewer's Source label) mapped to the `baselines.common.runner.Baseline` subclass that runs
it, written as "<module>:<class>".

The module is imported lazily, only when that baseline is actually asked for. Each baseline has
its own container with only its own dependencies installed, so importing them all up front would
fail everywhere — the price is that a typo in the module path only surfaces when that baseline
runs.
"""

import importlib

BASELINES = {
    "da3": "baselines.depth_anything_3.run_sequence:DepthAnything3",
    "da3_streaming": "baselines.da3_streaming.run_sequence:DA3Streaming",
    "vggt": "baselines.vggt.run_sequence:VGGT",
    "vggt-omega": "baselines.vggt-omega.run_sequence:VGGTOmega",
    "pi3": "baselines.Pi3.run_sequence:Pi3",
    "pi3x": "baselines.Pi3X.run_sequence:Pi3X",
    "vggt-slam": "baselines.VGGT-SLAM.run_sequence:VGGTSLAM",
    "vggt-long": "baselines.VGGT-Long.run_sequence:VGGTLong",
    "mpsfm": "baselines.mpsfm.run_sequence:MPSfM",
    "pi-long": "baselines.Pi-Long.run_sequence:PiLong",
    "colmap": "baselines.colmap.run_sequence:COLMAP",
    "glomap": "baselines.glomap.run_sequence:GLOMAP",
    "pixel-perfect-sfm": "baselines.pixel-perfect-sfm.run_sequence:PixelPerfectSfM",
    "vidmap": "baselines.vidmap.run_sequence:VidMap",
}


def available() -> list[str]:
    return sorted(BASELINES)


def get(name: str) -> type:
    """The `Baseline` subclass registered under `name`."""
    if name not in BASELINES:
        raise SystemExit(f"unknown baseline {name!r}; registered: {', '.join(available())}")
    module_path, _, class_name = BASELINES[name].partition(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    if cls.name != name:  # the key here and the class attribute pick the predictions/ folder
        raise SystemExit(f"{BASELINES[name]} calls itself {cls.name!r} but is registered as "
                         f"{name!r}; they must match")
    return cls
