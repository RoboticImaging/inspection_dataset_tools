"""Nan-safe reductions, so one hole in the data cannot nan out a whole row of a table.

Every metric here aggregates over data that is allowed to have holes: a depth pixel the method
declined to predict, a camera pair whose baseline is zero so the translation direction is
undefined, a point that found nothing to match against. Plain `mean`/`median` turn a single such
entry into a nan headline number, which is both wrong (the other 999,999 pixels were fine) and
silent (nan renders as "nan" in the table and nothing says why).

The rule throughout is: drop what is undefined, reduce over what is left, and let the caller say
what an empty result means via `default` — nan where the number is genuinely unknowable, 0.0
where "nothing qualified" is itself the answer. That is deliberately not `torch.nanmean`, which
returns nan for an all-nan slice regardless of which of those two cases it was, and reduces over
inf as if it were data.

Each of these takes tensors and returns a python float: a row of a results table is host-side
numbers, and this is the last device operation before it.
"""

import torch

from ..device import tensor


def finite(values) -> torch.Tensor:
    """The finite entries of `values`, flattened."""
    a = tensor(values).ravel()
    return a[torch.isfinite(a)]


def mean(values, default: float = float("nan")) -> float:
    a = finite(values)
    return float(a.mean()) if a.numel() else float(default)


def median(values, default: float = float("nan")) -> float:
    a = finite(values)
    if not a.numel():
        return float(default)
    # torch.median is the lower of the two middles on an even count; the average of them is what
    # numpy called the median here, and what the tables have always reported.
    ordered = a.sort().values
    n = ordered.numel()
    return float((ordered[(n - 1) // 2] + ordered[n // 2]) / 2)


def rms(values, default: float = float("nan")) -> float:
    a = finite(values)
    return float((a**2).mean().sqrt()) if a.numel() else float(default)


def fraction_below(values, threshold: float, default: float = float("nan")) -> float:
    """Fraction of the *defined* entries that are under `threshold`.

    Undefined entries leave the denominator rather than counting as failures: a pair of frames
    with no baseline is unscorable, not a miss. Where an undefined entry does mean a miss (a
    predicted point with no match at all), the caller passes `default=0.0` and the two readings
    coincide, because then every entry is undefined.
    """
    a = finite(values)
    return float(torch.count_nonzero(a < threshold) / a.numel()) if a.numel() else float(default)


def positive_depth(depth: torch.Tensor) -> torch.Tensor:
    """Mask of the pixels a depth map can be compared at: finite and in front of the camera.

    Depth metrics divide by depth and take its log, so zero and negative are as unusable as nan
    even though they survive `isfinite`.
    """
    d = tensor(depth)
    return torch.isfinite(d) & (d > 0)
