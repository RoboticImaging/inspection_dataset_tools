#!/usr/bin/env bash
# Build the DetectorFreeSfM baseline image. Run this once (and again after editing the Dockerfile,
# bumping the DetectorFreeSfM submodule, or changing third_party/RoIAlign.pytorch) — it does
# nothing else: no dataset paths, no running of the baseline.
#
# UNTESTED: this Dockerfile has not been built or run here yet (see its own header) — expect to
# iterate on real build errors the first few times this runs, the same way this repo's other
# from-source baselines (pixel-perfect-sfm, mpsfm) got to their current, validated state.
#
# Prerequisites, both on the host, before running this script:
#
#   1. Nested submodules — DetectorFreeSfM/third_party/RoIAlign.pytorch and
#      DetectorFreeSfM/third_party/Hierarchical_Localization are both required (the Dockerfile
#      COPYs the former in and builds it; the latter is imported directly by path at runtime from
#      the host mount) but are NOT initialized by default in this repo's checkout — an
#      uninitialized submodule copies as an empty directory and breaks this build with a missing
#      setup.py. Initialize them (third_party/multi-view-evaluation is optional — ETH3D
#      triangulation eval tooling only, not needed for a plain SfM run — include it too if you
#      want it):
#        git -C DetectorFreeSfM submodule update --init third_party/RoIAlign.pytorch third_party/Hierarchical_Localization
#
#   2. Pretrained weights — DetectorFreeSfM/INSTALL.md has you download a weight.tar from Google
#      Drive (linked partway down that file, before its "Modified COLMAP version installation"
#      section) and extract it at the DetectorFreeSfM repo root -- this can't be automated here
#      (interactive Google Drive download, no stable direct URL to `curl`/`wget`). Not needed to
#      build this image, but needed before this baseline can actually run -- it's read from the
#      host mount at runtime, same as the rest of DetectorFreeSfM/'s source (see the Dockerfile's
#      own header for why nothing here bakes the weight into the image). A REAL run confirmed this
#      produces MORE than just the one file hydra_configs/demo/dfsfm.yaml itself names
#      (`weight/multiview_matcher.ckpt`, the multi-view refinement matcher): the coarse matcher
#      -- `loftr_official` by default (baselines/configs/detector-free-sfm.yaml's own
#      `NEUSFM_coarse_matcher`) -- ALSO needs its own weight, hardcoded per-matcher in
#      src/coarse_match/coarse_match.py (NOT configurable via hydra overrides at all):
#        loftr_official -> weight/outdoor_ds.ckpt
#        aspanformer     -> weight/aspanformer_weights/outdoor.ckpt
#        matchformer     -> (see that same file for its own hardcoded path, if you switch to it)
#      A real run without weight/ downloaded yet fails exactly this way -- a plain
#      `FileNotFoundError: [Errno 2] No such file or directory: 'weight/outdoor_ds.ckpt'`, well
#      past every other fix in this file's own history (cv2/albumentations, torch/numpy, ...) --
#      confirming those are all actually resolved and this is the last remaining manual step.
#
# Usage:
#   baselines/DetectorFreeSfM/build_container.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/detector-free-sfm}"

if [ ! -f "${REPO_ROOT}/DetectorFreeSfM/third_party/RoIAlign.pytorch/setup.py" ]; then
    echo "error: DetectorFreeSfM/third_party/RoIAlign.pytorch looks uninitialized." >&2
    echo "  run: git -C ${REPO_ROOT}/DetectorFreeSfM submodule update --init third_party/RoIAlign.pytorch third_party/Hierarchical_Localization" >&2
    exit 1
fi

# Build context is the repo root: the Dockerfile COPYs in DetectorFreeSfM/requirements.txt and
# DetectorFreeSfM/third_party/RoIAlign.pytorch only (not the whole submodule — see the Dockerfile's
# own header for why).
docker build -f "${REPO_ROOT}/baselines/DetectorFreeSfM/Dockerfile" \
    -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — try it with, e.g.:"
echo "  docker run -it --rm --gpus all -v \"\$PWD\":/working -w /working $IMAGE bash"
