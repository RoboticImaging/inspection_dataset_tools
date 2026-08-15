"""Where the arrays live.

Every array in this package is a `torch.Tensor` on one device, resolved once here: cuda when the
machine has it, cpu otherwise, or whatever `INSPECTION_EVAL_DEVICE` / `use()` says. Data still
arrives as numpy — an EXR depth map, a `.ply`, a mesh sampled by trimesh, a Blender cam json —
and leaves as python floats in a results row, so `tensor()` and `numpy()` are the two boundary
crossings and nothing else in the package needs to know what a device is.

`tensor()` keeps an existing tensor's dtype and only moves it, so a float32 cloud stays float32
(the cloud metric's nearest-neighbour search is the one thing here big enough to care), while
anything coming in from numpy or python is read at `DTYPE` — float64, the dtype numpy would have
used, so the pose and alignment numbers are the same ones the original code produced.
"""

import os

import numpy as np
import torch

DTYPE = torch.float64


def resolve(name: str | None = None) -> torch.device:
    """The device to evaluate on: `name`, else `$INSPECTION_EVAL_DEVICE`, else cuda if there is
    one. An explicitly asked-for cuda that isn't there is an error rather than a silent, and very
    slow, fallback to the cpu — and it is raised here, where it can say so, rather than out of
    the first tensor a metric happens to make."""
    name = name or os.environ.get("INSPECTION_EVAL_DEVICE")
    if not name:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"asked for {name!r}, but torch reports no cuda device")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"asked for {name!r}, but this machine has "
                               f"{torch.cuda.device_count()} cuda device(s)")
    return device


DEVICE = resolve()


def use(name: str | None) -> torch.device:
    """Point the package at another device. Call it before anything else runs — tensors already
    made stay where they were made."""
    global DEVICE
    DEVICE = resolve(name)
    return DEVICE


def tensor(values, dtype: torch.dtype | None = None) -> torch.Tensor:
    """`values` on the eval device: a tensor is moved, anything else is read through numpy."""
    if isinstance(values, torch.Tensor):
        return values.to(device=DEVICE, dtype=dtype)
    return torch.as_tensor(np.asarray(values), dtype=dtype or DTYPE).to(DEVICE)


def stack(values, dtype: torch.dtype | None = None) -> torch.Tensor:
    """One tensor out of a sequence of same-shaped arrays, whatever they each are now."""
    return torch.stack([tensor(v, dtype) for v in values])


def numpy(values) -> np.ndarray:
    """Back to the host, for the few libraries that only speak numpy (trimesh, cv2, np.save)."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)
