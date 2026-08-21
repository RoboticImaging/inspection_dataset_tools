"""The metric registry.

A metric is a function `(EvalCase, options) -> {column: value}` plus a declaration of what data
it needs and what knobs it takes. Registering it is the whole integration: the runner discovers
it, skips it on cases that lack the data, exposes its options as `--set <metric>.<key>=<value>`,
and gives it a table of its own whose columns are whatever keys it returned.

Adding a metric, in full:

    @dataclass
    class MyOptions:
        threshold: float = 0.1

    @register("my_metric", needs=(CAP_CLOUD,), options=MyOptions)
    def my_metric(case: EvalCase, o: MyOptions) -> dict:
        return {"My number": ...}

...then import the module in `metrics/__init__.py` so the decorator runs.

`columns` is an optional extra: a `(options) -> list[str]` function naming, in order, the keys
`fn` would return for those options -- without touching a `case` at all. Every column name
`fn` produces already has to be a pure function of `o` (never of the data) for this to be
correct, which is true of every metric so far (thresholds and the like only ever pick *which*
keys appear, not what data goes into them). It exists for one narrow reason: when a run's
target hit a GPU OOM, the runner never gets to call `fn` and so has no other way to learn what
columns that case's row should have had (see `runner.evaluate`'s OOM handling). A metric that
skips `columns` just keeps today's behaviour: an all-OOM run's table for it has only the label
columns.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    needs: tuple = ()
    options: type = None  # a dataclass of knobs, instantiated fresh per run
    columns: Callable = None  # optional: (options) -> list[str], see module docstring

    def default_options(self):
        return self.options() if self.options is not None else None


REGISTRY: dict[str, Metric] = {}


def register(name: str, needs: tuple = (), options: type = None, columns: Callable = None):
    def wrap(fn):
        if name in REGISTRY:
            raise ValueError(f"metric {name!r} is already registered")
        REGISTRY[name] = Metric(name, fn, tuple(needs), options, columns)
        return fn
    return wrap


def get(name: str) -> Metric:
    if name not in REGISTRY:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def apply_overrides(options, overrides: dict):
    """Set `options` fields from strings (`--set depth.scale=median`), typed by the annotation."""
    if not overrides:
        return options
    fields = {f.name: f for f in dataclasses.fields(options)}
    for key, text in overrides.items():
        if key not in fields:
            raise KeyError(f"unknown option {key!r}; known: {sorted(fields)}")
        setattr(options, key, _coerce(fields[key].type, text))
    return options


def _coerce(annotation, text: str):
    ann = str(annotation)
    if text.lower() in ("none", "null") and "None" in ann:
        return None
    if "tuple" in ann or "," in text and "str" not in ann:
        return tuple(_number(part) for part in text.split(","))
    if "bool" in ann:
        return text.strip().lower() in ("1", "true", "yes", "on")
    if "int" in ann:
        return int(text)
    if "float" in ann:
        return float(text)
    return text


def _number(text: str):
    try:
        return int(text)
    except ValueError:
        return float(text)
