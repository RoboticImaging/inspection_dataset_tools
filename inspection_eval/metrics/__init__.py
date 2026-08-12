"""Metric registry, plus every built-in metric. Import a new metric module here to register it."""

from .base import REGISTRY, Metric, apply_overrides, get, register
from . import cloud, depth, poses  # noqa: F401  (imported for their side effect: registration)

__all__ = ["REGISTRY", "Metric", "apply_overrides", "get", "register"]
