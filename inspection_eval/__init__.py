"""Evaluation of baseline predictions against the dataset's ground truth.

Point it at a prediction folder (`<asset>/predictions/<method>`) and it scores every parameter
combination that method was run on, producing one results table per metric:

    from inspection_eval import evaluate, target_from_path
    results = evaluate([target_from_path("/data/old_house/predictions/da3")])
    print(results.tables["pose"].to_markdown())

`inspection_eval.case.EvalCase` is the layer that pairs gt with a prediction and puts the
prediction in the gt gauge; `inspection_eval.metrics` is the registry of scoring functions
(see `metrics/base.py` for how to add one).
"""

from .align import Sim3, umeyama
from .case import EvalCase
from .metrics import REGISTRY, Metric, register
from .report import Table
from .runner import Results, Target, evaluate, target_from_path, targets_from_paths, variants

__all__ = ["EvalCase", "Metric", "REGISTRY", "Results", "Sim3", "Table", "Target",
           "evaluate", "register", "target_from_path", "targets_from_paths", "umeyama",
           "variants"]
