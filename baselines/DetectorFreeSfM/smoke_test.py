"""Fast smoke test for the DetectorFreeSfM baseline's environment -- catches the whole class of
bugs this baseline has hit so far (incompatible pip packages, missing/wrong-arch compiled CUDA
extensions, wrong hardcoded weight paths, a forked `colmap` CLI whose flags silently drifted from
what run_sequence.py assumes) in well under a minute, without a GPU and without a real end-to-end
run (which takes 20+ minutes to reach the first failure, more past it).

Run inside the container (build_container.sh once, no run_container.sh needed -- this doesn't
touch any host dataset):
    docker run --rm inspection-baselines/detector-free-sfm \
        python /path/to/repo/baselines/DetectorFreeSfM/smoke_test.py
or, from a shell (run_container.sh):
    python smoke_test.py

WHAT THIS CATCHES, and how:
  - Import-time errors (ModuleNotFoundError, AttributeError from an incompatible library version
    -- this baseline's own real history: timm, albumentations/qudida, numpy/opencv) -- by actually
    importing and constructing every model this baseline's real code path touches, not just
    `import`ing top-level packages.
  - Missing or wrong-arch-built compiled CUDA extensions (this baseline's own real history:
    third_party/RoIAlign.pytorch's `crop_and_resize_gpu`, silently never built because its own
    setup.py -- not just this Dockerfile's TORCH_CUDA_ARCH_LIST -- separately gates on
    `torch.cuda.is_available()` at BUILD time, always False under `docker build`) -- by patching
    `torch.cuda.is_available()` to True for this process only (see `_patch_torch_cuda` below),
    which forces every `if torch.cuda.is_available(): import ...`-style branch in this codebase
    (and in roi_align's own identical pattern) to actually execute, on any machine, GPU or not.
  - Wrong/hardcoded weight paths (this baseline's own real history: `weight/outdoor_ds.ckpt` isn't
    named anywhere in hydra_configs/demo/dfsfm.yaml -- it's hardcoded per-matcher in
    src/coarse_match/coarse_match.py) -- by actually loading every checkpoint via each matcher's
    own real `build_model`/checkpoint-loading code, not guessing the paths ourselves.
  - The modified `colmap` CLI's dense-stage flags silently drifting from what
    run_sequence.py's own PatchMatchConfig/StereoFusionConfig assume (this baseline's own real
    history: an earlier version of those dataclasses was copied from baselines/colmap's own,
    unrelated `colmap` build, and included a --PatchMatchStereo.num_threads flag this fork's
    binary doesn't have at all) -- by parsing `colmap patch_match_stereo --help` / `colmap
    stereo_fusion --help` directly and diffing the flag names against those two dataclasses.
  - An actual coarse-matching forward pass (LoFTR/ASpanFormer/MatchFormer) on two tiny synthetic
    images, on CPU (`torch.Tensor.cuda`/`torch.nn.Module.cuda` are patched to no-ops -- see below)
    -- catching gross shape/API mismatches in the real code path, not just import errors.

WHAT THIS DOES NOT CATCH: anything that only manifests on real image content or at real scale
(registration/triangulation quality, bundle-adjustment convergence, actual CUDA kernel numerics),
GPU-specific runtime failures (OOM, a real device's own driver quirks), or whether the dense
stage's real *output* is sane (only that its CLI *accepts* the flags this baseline sends it).
Those still need a real end-to-end run_sequence.py run on a GPU machine.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

DFSFM_ROOT = Path(__file__).resolve().parents[2] / "DetectorFreeSfM"

FAILURES: list[str] = []


def _report(name: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _patch_torch_cuda() -> None:
    """Makes `torch.cuda.is_available()` report True (so every GPU-conditional import/branch in
    this codebase and in roi_align's own identical pattern actually runs -- see module docstring)
    and `.cuda()` a CPU-staying no-op (so `detector.cuda(); matcher.cuda()`-style calls in this
    codebase, unconditional in its own source, don't crash without a real device). Scoped to this
    process only -- nothing outside this script is affected."""
    import torch

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda *a, **k: "smoke-test-fake-cuda"
    torch.cuda.get_device_capability = lambda *a, **k: (8, 6)
    torch.cuda.empty_cache = lambda: None
    torch.cuda.synchronize = lambda *a, **k: None
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self


def _make_synthetic_pair(dir_: Path) -> tuple[Path, Path]:
    """Two tiny (256x256) random grayscale-ish images -- real files on disk, real matcher forward
    pass, just not real content (no correct matches are expected; this only checks nothing
    crashes)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        p = dir_ / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths[0], paths[1]


def _run_coarse_matcher_inprocess(matcher: str, tmp_root: Path) -> None:
    """The actual work check_coarse_matcher does -- factored out so it can run either in this
    process (fine for exactly one matcher) or, via `--_internal-single-matcher`, in a fresh
    subprocess (see check_coarse_matcher's own docstring for why more than one matcher needs
    that)."""
    from src.coarse_match.coarse_match import detector_free_coarse_matching

    work = tmp_root / matcher
    work.mkdir(parents=True)
    img0, img1 = _make_synthetic_pair(work)
    detector_free_coarse_matching(
        [str(img0), str(img1)],
        [f"{img0} {img1}"],
        feature_out=str(work / "keypoints.h5"),
        match_out=str(work / "matches.h5"),
        img_resize=256,
        matcher=matcher,
        match_type="coarse_only",
        use_ray=False,
        verbose=False,
    )


def check_coarse_matcher(matcher: str) -> None:
    """Runs DetectorFreeSfM's own real `detector_free_coarse_matching` (src/coarse_match/
    coarse_match.py) end to end -- model construction, checkpoint loading, and one real forward
    pass -- on two synthetic images, for one `NEUSFM_coarse_matcher` choice.

    Runs in an ISOLATED subprocess (`python smoke_test.py --_internal-single-matcher <matcher>`),
    not in this process -- found necessary by actually running this script: LoFTR's own
    third_party/LoFTR/configs/loftr/outdoor/loftr_ds_coarse_only.py and ASPanFormer's own
    third_party/aspantransformer/configs/aspan/outdoor/aspan_test_coarse_only.py both do
    `sys.path.append(...); from default import _CN as cfg` -- importing their own per-matcher
    yacs config module under the SAME generic top-level name "default" (not their real dotted
    package path). Python's own `sys.modules` caches whichever one imports first under that name
    for the rest of the process; a second matcher's `from default import ...` in the same process
    then silently gets the FIRST matcher's `_CN` object instead of its own, which has no
    `.ASPAN` (or whichever) attribute -- real error: "AttributeError: ASPAN", hit here by testing
    loftr_official immediately before aspanformer. This is a real, latent quirk in those two
    third-party configs, but NOT one a real run of this baseline ever hits: run_sequence.py's own
    `run_dfsfm` subprocesses `eval_dataset.py` once per sequence, always with exactly one
    `NEUSFM_coarse_matcher` for that whole process -- multiple matchers are never loaded in the
    same interpreter outside of this smoke test's own attempt to check all of them quickly.
    Isolating each in its own subprocess avoids the false failure this collision would otherwise
    cause, and is also simply more faithful to how these matchers are ever really invoked."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_internal-single-matcher", matcher],
        cwd=DFSFM_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")


def check_multiview_matcher() -> None:
    """Constructs the post-optimization refinement matcher (src/post_optimization/matcher_model/
    multiview_match_worker.py's own `build_model`) and loads its real checkpoint -- the exact
    import chain (MultiviewMatcher -> backbone -> S2DNet -> roi_align) that broke on this
    baseline's own real history. Construction + checkpoint load only, no forward pass (needs real
    point tracks from an actual reconstruction to synthesize meaningfully)."""
    from src.post_optimization.matcher_model.multiview_match_worker import build_model

    build_model({
        "cfg_path": ["hydra_training_configs/experiment/multiview_refinement_matching.yaml"],
        "weight_path": ["weight/multiview_matcher.ckpt"],
        "seed": 666,
    })


def check_roi_align_gpu_extension() -> None:
    """Direct, standalone check for the exact compiled extension this baseline's own real history
    found missing -- redundant with check_multiview_matcher's own import chain (which reaches the
    same module), but isolates the failure to one clear cause if it ever regresses, rather than a
    deep traceback through six unrelated files."""
    import roi_align.crop_and_resize_gpu  # noqa: F401


def check_colmap_dense_cli_flags() -> None:
    """Diffs run_sequence.py's own PatchMatchConfig/StereoFusionConfig field names against the
    modified `colmap` binary's own `--help` output for patch_match_stereo/stereo_fusion -- catches
    a dataclass drifting from what this fork's CLI actually accepts (this baseline's own real
    history: an earlier PatchMatchConfig had a --PatchMatchStereo.num_threads field this fork's
    binary has never had) without needing a GPU or a real reconstruction to run either command
    against."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_sequence import PatchMatchConfig, StereoFusionConfig  # noqa: E402

    for command, prefix, cfg_cls in (
        ("patch_match_stereo", "PatchMatchStereo", PatchMatchConfig),
        ("stereo_fusion", "StereoFusion", StereoFusionConfig),
    ):
        proc = subprocess.run(["colmap", command, "--help"], capture_output=True, text=True)
        cli_flags = {
            line.split()[0].removeprefix(f"--{prefix}.")
            for line in proc.stdout.splitlines()
            if line.strip().startswith(f"--{prefix}.")
        }
        cfg_fields = {f.name for f in dataclasses.fields(cfg_cls)}
        missing_from_cfg = cli_flags - cfg_fields   # colmap has it, our dataclass doesn't
        extra_in_cfg = cfg_fields - cli_flags        # our dataclass has it, colmap doesn't (this
                                                      # is the one that actually crashes a run)
        if extra_in_cfg:
            raise AssertionError(
                f"{cfg_cls.__name__} has field(s) {sorted(extra_in_cfg)} that `colmap {command} "
                f"--help` doesn't recognize -- a real dense-MVS run would fail with an "
                f"'unrecognised option' error on these"
            )
        if missing_from_cfg:
            print(f"  note: `colmap {command} --help` has flag(s) {sorted(missing_from_cfg)} "
                  f"that {cfg_cls.__name__} doesn't expose -- not a crash (colmap applies its own "
                  f"default), just a knob this baseline can't currently tune")


def _run_single_matcher_subprocess_entrypoint(matcher: str) -> int:
    """Entry point for the isolated `python smoke_test.py --_internal-single-matcher <matcher>`
    subprocess check_coarse_matcher spawns -- see that function's own docstring for why this needs
    to be a fresh process rather than an in-process call. Not part of the normal `main()` flow."""
    sys.path.insert(0, str(DFSFM_ROOT))
    _patch_torch_cuda()
    with tempfile.TemporaryDirectory(prefix="dfsfm-smoke-single-") as tmp_str:
        try:
            _run_coarse_matcher_inprocess(matcher, Path(tmp_str))
        except Exception:
            traceback.print_exc()
            return 1
    return 0


def main() -> int:
    if not DFSFM_ROOT.is_dir():
        print(f"error: {DFSFM_ROOT} not found -- run this inside the container, "
             "from anywhere under the repo checkout", file=sys.stderr)
        return 2

    import os
    os.chdir(DFSFM_ROOT)
    sys.path.insert(0, str(DFSFM_ROOT))

    _patch_torch_cuda()

    weight_files = [
        "weight/outdoor_ds.ckpt", "weight/aspanformer_weights/outdoor.ckpt",
        "weight/multiview_matcher.ckpt",
    ]
    missing_weights = [w for w in weight_files if not (DFSFM_ROOT / w).is_file()]
    if missing_weights:
        print(f"error: missing {missing_weights} under {DFSFM_ROOT} -- download weight.tar per "
             "INSTALL.md first (see build_container.sh's own header)", file=sys.stderr)
        return 2

    checks = [
        ("roi_align GPU extension", check_roi_align_gpu_extension),
        ("colmap dense-stage CLI flags", check_colmap_dense_cli_flags),
        ("coarse matcher: loftr_official", lambda: check_coarse_matcher("loftr_official")),
        ("coarse matcher: aspanformer", lambda: check_coarse_matcher("aspanformer")),
        ("coarse matcher: matchformer", lambda: check_coarse_matcher("matchformer")),
        ("multiview refinement matcher", check_multiview_matcher),
    ]

    for name, fn in checks:
        try:
            fn()
            _report(name, True)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: report every check
            _report(name, False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{len(checks)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_internal-single-matcher":
        sys.exit(_run_single_matcher_subprocess_entrypoint(sys.argv[2]))
    sys.exit(main())
