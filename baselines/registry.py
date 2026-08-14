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
}


def available() -> list[str]:
    return sorted(BASELINES)


def get(name: str) -> type:
    """The `Baseline` subclass registered under `name`."""
    if name not in BASELINES:
        raise SystemExit(f"unknown baseline {name!r}; registered: {', '.join(available())}")
    module_path, _, class_name = BASELINES[name].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
