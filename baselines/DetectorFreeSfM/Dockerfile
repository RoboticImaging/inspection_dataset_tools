# syntax=docker/dockerfile:1
#
# Detector-Free SfM baseline (zju3dv/DetectorFreeSfM, CVPR 2024) -- LoFTR/ASpanFormer/MatchFormer
# coarse matching + a learned multi-view refinement matcher, feeding a MODIFIED COLMAP fork that
# adds a feature-metric keypoint/geometry refinement stage (see DetectorFreeSfM/INSTALL.md's own
# "Modified COLMAP version installation" section).
#
# HONESTY NOTE, unlike this repo's `pixel-perfect-sfm/Dockerfile`: this file has NOT been built or
# run end-to-end here -- there is no validated image behind it yet, no GPU run, nothing. It is a
# direct, best-effort translation of DetectorFreeSfM/INSTALL.md and DetectorFreeSfM/environment.yaml
# into this repo's own Dockerfile conventions (see baselines/pixel-perfect-sfm/Dockerfile and
# baselines/mpsfm/Dockerfile for the conventions copied here). Treat every version pin and flag
# below as "upstream's own documented recipe, adapted" rather than "proven on real hardware" --
# run build_container.sh and fix forward from whatever actually breaks, the same way this repo's
# other from-source baselines got to their current, validated state.
#
# Base image / CUDA version: reuses baselines/pixel-perfect-sfm's own choice of
# nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 rather than picking a fresh one, for two independent
# reasons that happen to agree here: (1) DetectorFreeSfM's own environment.yaml pins
# `pytorch=1.8.0` + `torchvision=0.9.1`, a pairing whose only official CUDA-11-class wheels are the
# `+cu111` ones on download.pytorch.org -- so CUDA 11.1 is the natural match for the pinned torch
# version itself, independent of COLMAP; and (2) the modified COLMAP fork below
# (github.com/hxy-123/colmap) is old-style SiftGPU/CUDA code using the legacy texture-reference API
# (`cudaBindTexture`) -- DetectorFreeSfM's own issue tracker (#18, #29) shows it failing to compile
# against newer CUDA/driver combos (deprecated texture API warnings on Ubuntu 22, and an
# `Unsupported gpu architecture 'compute_35'` hard-fail against CUDA 12.1 on Kaggle) -- i.e. the
# same "needs an old-CUDA toolchain" shape as vanilla COLMAP 3.8 did for pixel-perfect-sfm.
# baselines/pixel-perfect-sfm/Dockerfile already validated (real build + GPU runs, see its own
# header) that this exact base image's nvcc, given `-DCMAKE_CUDA_ARCHITECTURES=89` explicitly,
# compiles COLMAP-vintage CUDA code that runs correctly on this repo's actual GPUs (2x RTX 6000 Ada
# Generation, compute 8.9) despite predating official sm_89 support -- reused here on the
# (untested for THIS fork specifically) assumption that a sibling old-COLMAP CUDA codebase behaves
# the same way. The `torch==1.8.0+cu111` wheel itself running correctly on compute 8.9 via PTX
# forward-compatibility is a SEPARATE, also-untested assumption -- COLMAP's own C++/CUDA build and
# PyTorch's own prebuilt-wheel kernels are compiled by entirely different toolchains, so
# pixel-perfect-sfm's validation says nothing about this one.
#
# Eigen: apt's `libeigen3-dev` (Ubuntu 20.04: 3.3.7), exactly as INSTALL.md's own apt-get list
# says -- NOT built from source the way baselines/pixel-perfect-sfm/Dockerfile does. That file
# needed Eigen 3.4.0 from source specifically because Ubuntu 22.04's packaged Eigen was one minor
# version below what pyceres/pixsfm's bindings needed; DetectorFreeSfM's own INSTALL.md, targeting
# an older Ceres (1.14.x, below) against plain apt Eigen on (implicitly) an older Ubuntu, documents
# no such requirement, so there's no known reason to deviate from what upstream itself specifies.
#
# Ceres: INSTALL.md pins branch `1.14.x` explicitly (much older than pixel-perfect-sfm's Ceres
# 2.1.0 -- these are two unrelated C++ codebases with independent, non-interchangeable pins; do not
# "align" this with the other baseline's Ceres version).
#
# Modified COLMAP: `git clone https://github.com/hxy-123/colmap.git` per INSTALL.md, no branch/tag
# given upstream (uses whatever its default branch is at clone time) -- built with
# `-DCMAKE_CUDA_ARCHITECTURES=89` for the same real-hardware reason as pixel-perfect-sfm's own
# COLMAP build (see above); INSTALL.md's own bare `cmake ..` doesn't set this at all, which may or
# may not matter for this fork's own (older-vintage, possibly pre-`CMAKE_CUDA_ARCHITECTURES`) CMake
# setup -- harmless to pass either way. `libcgal-qt5-dev` is called out as a second, separate
# `apt-get install` in INSTALL.md (not merged into the main list) -- kept as its own line below for
# the same reason pixel-perfect-sfm's own header cautions against silently "cleaning up" an
# upstream recipe's odd shape without a reason to believe it doesn't matter.
#
# Python: 3.9 via the deadsnakes PPA, matching environment.yaml's own `python=3.9` pin exactly
# (Ubuntu 20.04's own default `python3` is 3.8) -- confirmed (PyPI file listing, not assumed) that
# both `opencv_python==4.4.0.46` and `pycolmap==0.3.0`, the two pins in requirements.txt most likely
# to lack a cp39 wheel at that vintage, DO ship manylinux2014 cp39 wheels, so this pin doesn't force
# any from-source Python package builds.
#
# `pycolmap==0.3.0` here is the plain PyPI package (bundles its own prebuilt COLMAP internally, cp39
# manylinux wheel, confirmed above) -- unrelated to, and not in conflict with, the hxy-123/colmap
# fork built from source above (that one only provides the system `colmap` CLI binary that
# DetectorFreeSfM/src/sfm_runner/sfm_model_geometry_refiner.py invokes via `subprocess`, reading
# `$COLMAP_PATH`, default `colmap` off PATH -- `make install` below puts it there, so no env var
# needs setting).
#
# third_party/RoIAlign.pytorch (INSTALL.md: "used in our multi-view refinement matching phase") is
# a required nested submodule of the DetectorFreeSfM submodule itself, and is copied in and built
# below -- NOT initialized by default in this repo's checkout (`git submodule status --recursive --
# DetectorFreeSfM` shows it, and `third_party/Hierarchical_Localization` -- also required, imported
# directly by path as `third_party.Hierarchical_Localization.hloc...`, source-only, no pip install
# needed for it -- both prefixed `-`, i.e. uninitialized). See build_container.sh's own header for
# the exact `git submodule update --init` command needed before this Dockerfile can build.
# `third_party/multi-view-evaluation` is NOT required here -- it's only ETH3D triangulation-metric
# tooling (INSTALL.md's own separate, optional install step), not imported by eval_dataset.py's
# main SfM path.
#
# Unlike baselines/pixel-perfect-sfm, this Dockerfile does NOT `COPY` the whole DetectorFreeSfM/
# tree in and build anything in place inside it -- there's no compiled extension living inside that
# tree the way pixsfm's own `.so` does (RoIAlign's compiled extension installs into the venv's
# site-packages via a plain, non-editable `pip install .`, fully decoupled from its source dir
# afterwards). So only the two subtrees actually needed at build time are COPY'd in below
# (DetectorFreeSfM/requirements.txt and DetectorFreeSfM/third_party/RoIAlign.pytorch); the rest of
# DetectorFreeSfM/ -- src/, hydra_configs/, eval_dataset.py, and (separately, see build_container.sh)
# the manually-downloaded weight/ dir and SfM_dataset/ -- is read directly off the host mount by
# run_container.sh instead (same "no rebuild needed after editing" convention as baselines/mpsfm),
# not baked into the image.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# INSTALL.md's own "Modified COLMAP version installation" apt list verbatim, plus: `curl` and
# `wget` (`git` is already in INSTALL.md's list but called out here since it's also used for
# Ceres/COLMAP clones), the deadsnakes PPA machinery for Python 3.9 (see header), and `libgl1` --
# requirements.txt pins the non-headless `opencv_python` (not `opencv-python-headless`, unlike
# this repo's own top-level requirements.txt), which dynamically links against libGL at import
# time; this base image has no GUI stack installed by default and `import cv2` would otherwise
# fail with "libGL.so.1: cannot open shared object file". `libeigen3-dev` IS kept here (unlike
# pixel-perfect-sfm/Dockerfile) -- see header.
#
# `wget` specifically: NOT just "general-purpose/kept for parity" (an earlier version of this
# comment said so, and dropped it on that basis) -- src/MultiviewMatcher/backbone/S2DNet/
# s2dnet.py's own `_init()` shells out to the `wget` CLI directly (`subprocess.call(["wget",
# self.url, "-q"], ...)`) to lazily fetch its own weights on first use, unconditionally, whenever
# S2DNet is the backbone in play (confirmed by smoke_test.py's own multiview-refinement-matcher
# check, which constructs exactly this backbone: real error without `wget` installed,
# "FileNotFoundError: [Errno 2] No such file or directory: 'wget'"). This is a genuine RUNTIME
# dependency of this baseline's own code, not a build-time convenience -- must stay installed.
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        git \
        cmake \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libfreeimage-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libmetis-dev \
        libcgal-qt5-dev \
        libatlas-base-dev \
        curl \
        wget \
        libgl1 \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

# Ceres 1.14.x, per INSTALL.md exactly (do not "helpfully" bump this to match pixel-perfect-sfm's
# unrelated 2.1.0 pin -- see header). BUILD_TESTING/BUILD_EXAMPLES=OFF for build time only, same as
# every other from-source Ceres build in this repo.
WORKDIR /opt
RUN git clone --branch 1.14.x --depth 1 https://github.com/ceres-solver/ceres-solver.git
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && make -j && make install

# Modified COLMAP fork, per INSTALL.md's own clone URL. -DCMAKE_CUDA_ARCHITECTURES=89 is this
# repo's own addition (this repo's real GPUs, see header) -- not present in INSTALL.md's own bare
# `cmake ..`.
WORKDIR /opt
RUN git clone https://github.com/hxy-123/colmap.git
WORKDIR /opt/colmap/build
RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 && make -j && make install

# Python 3.9 venv (repo convention -- keeps `python`/`pip` pointed at this image's own install
# regardless of what a host ~/.bashrc bind-mounted in by run_container.sh might inject onto PATH).
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3.9 -m venv /opt/venv && pip install --no-cache-dir -U pip wheel

# torch==1.8.1 / torchvision==0.9.1, as their `+cu111` wheel variants -- the CUDA-11-class build
# matching this image's own CUDA 11.1.1 (see header; environment.yaml's own `cudatoolkit=11.7`
# line is explicitly marked "# Modify here!" by upstream itself, i.e. meant to be adapted to
# whatever CUDA the build host actually has, not followed literally -- there is no published
# `torch==1.8.0`+cu117 combination to follow literally anyway).
#
# torch bumped 1.8.0 -> 1.8.1 from environment.yaml's own literal pin: a REAL build of this file
# failed here -- pip's own resolver error was explicit and unambiguous: "torchvision 0.9.1+cu111
# depends on torch==1.8.1" (torchvision wheels pin an exact torch version each release; 0.9.1 was
# built against 1.8.1, not 1.8.0 -- environment.yaml's own torch=1.8.0/torchvision=0.9.1 pair was
# never actually installable together as stated, on pip or conda). Bumped torch's patch version
# (1.8.0 -> 1.8.1, a bugfix-only release) rather than downgrading torchvision to 0.9.0 (which
# would pair correctly with 1.8.0): torch's own version matters far more to everything downstream
# of it in this file (RoIAlign's CUDA extension, kornia, pytorch-lightning) than torchvision's
# does, so keeping requirements.txt's own torchvision==0.9.1 exact and moving the one-patch-level
# torch pin instead is the smaller, safer deviation from upstream's stated versions.
#
# `numpy<2` constraint, applied globally from here on (not just this one RUN): a REAL build also
# failed downstream of this step (building third_party/RoIAlign.pytorch below) with
# "AttributeError: _ARRAY_API not found" / "A module that was compiled using NumPy 1.x cannot be
# run in NumPy 2.0.2" -- torch==1.8.1's own compiled `torch._C` extension (2021-era) was built
# against NumPy 1.x's C API, and torch's own unpinned `numpy` dependency let pip resolve the
# newest NumPy (2.0.2) instead. Rather than pinning `numpy<2` only on this one `pip install`
# (leaving it free to be upgraded back to 2.x by any later, unrelated install -- e.g. this file's
# own inspection_viz-deps step further down, which installs viser), PIP_CONSTRAINT applies this
# bound to every `pip install` for the rest of this build, the same "fix it once, globally" shape
# as ENV VIRTUAL_ENV above.
# `timm==0.4.12` constraint, added to the same file for the same "fix it once, globally" reason:
# requirements.txt's own `timm>=0.3.2` is a loose lower bound only, so it resolved a current timm
# release -- a REAL run (well past everything above: through coarse LoFTR matching and the coarse
# COLMAP mapper, into post-optimization) failed at `from timm.models.registry import
# register_model` (src/MultiviewMatcher/backbone/resnet.py) with "AttributeError: module 'torch'
# has no attribute 'fx'" -- that current timm release's own `timm/layers/attention.py` uses
# `@torch.fx.wrap` at *module import time*, a torch.fx usage pattern this old torch==1.8.1 doesn't
# support the way timm now assumes. 0.4.12 confirmed (real container test, not assumed) to import
# clean against torch==1.8.1, and PyPI's own metadata for that release shows its only two
# dependencies are torch/torchvision (both already satisfied here) -- so no --no-deps/hand-picked
# replacements needed here, unlike albumentations/qudida below.
RUN printf 'numpy<2\ntimm==0.4.12\n' > /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt
RUN pip install --no-cache-dir \
        torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Remainder of environment.yaml's conda deps, translated to pip (tqdm/matplotlib/scipy have no
# native/compiled requirement forcing conda specifically; pylint/ipdb are dev tools upstream
# happens to list here too).
RUN pip install --no-cache-dir \
        tqdm matplotlib pylint scipy ipdb \
        h5py==3.1.0 loguru==0.5.3

WORKDIR /DetectorFreeSfM

# This repo's own pinned requirements.txt (pycolmap, hydra-core, kornia, pytorch-lightning, ray,
# ...) -- see header for the pycolmap==0.3.0 note.
COPY DetectorFreeSfM/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# `albumentations`, NOT in DetectorFreeSfM's own requirements.txt at all, despite being imported
# unconditionally by src/dataset/utils.py (reached from eval_dataset.py's own main
# coarse-matching path, `from .coarse_match.coarse_match import detector_free_coarse_matching`
# -> ... -> `import albumentations as A`). A REAL run of this baseline failed here with a plain
# `ModuleNotFoundError: No module named 'albumentations'` -- a genuine gap in upstream's own
# requirements.txt, not something this file was previously getting wrong. If another
# `ModuleNotFoundError` surfaces the same way on a later run, it belongs here too, with the same
# kind of comment -- upstream's requirements.txt has already been shown to be incomplete once.
#
# Pinned to 1.3.1 (confirmed via `grep -rn albumentations DetectorFreeSfM/src` that `A` -- the
# only name bound by that import -- is never actually referenced anywhere else in this whole
# submodule; this is a dead import, so any version that merely IMPORTS cleanly is functionally
# equivalent here) -- specifically the last release before albumentations' own 2024 refactor that
# split pixel-level ops into a separate `albucore` package: a REAL attempt at plain, unpinned
# `pip install albumentations` here resolved to a current release that then failed at *runtime*,
# elsewhere, on a plain `import albumentations` with `ModuleNotFoundError: No module named
# 'albucore'` (that release's own `augmentations/pixel/functional.py` imports it unconditionally,
# and this baseline deliberately doesn't pip-install every one of albumentations' own transitive
# deps -- see --no-deps below). 1.3.1 predates that split and imports clean with nothing beyond
# what's installed by hand below.
#
# --no-deps, then albumentations' real deps by hand (numpy/scipy already installed above): a
# plain, dependency-following `pip install albumentations` here builds fine but breaks at
# RUNTIME, later, at a plain `import cv2` anywhere in this repo's own inspection_viz -- real
# error: "AttributeError: partially initialized module 'cv2' has no attribute
# '_registerMatType' (most likely due to a circular import)". `_registerMatType` is from a MUCH
# newer OpenCV than requirements.txt's own `opencv_python==4.4.0.46` -- albumentations' own
# setup.py tries to detect an already-installed opencv-python and skip adding its own
# `opencv-python-headless` dependency, but that detection is well documented as unreliable in a
# plain `pip install` (see albumentations-team/albumentations issues #841, #697, #1043) -- here it
# installed opencv-python-headless anyway, a SEPARATE PyPI distribution that also provides the
# `cv2` package, and its files landed in site-packages/cv2/ alongside opencv_python==4.4.0.46's
# own, corrupting the package. --no-deps sidesteps the detection entirely rather than trusting it.
#
# `qudida` is included pre-emptively (many albumentations releases, 1.3.1 included, import it
# unconditionally at package-init time, via `augmentations/domain_adaptation.py`) -- also
# --no-deps, for the SAME reason as albumentations itself: a first version of this fix installed
# it plainly, on the assumption it only needed numpy/scipy (already present), and that
# reintroduced the exact same opencv collision one level removed -- `pip show
# opencv-python-headless` in the built image confirmed "Required-by: albumentations, qudida"
# (PyPI's own qudida metadata: it depends directly on `opencv-python-headless>=4.0.1`, not just on
# scikit-learn as assumed). scikit-learn is qudida's own other real dependency, added by hand in
# its place. All of this -- --no-deps twice, three hand-picked replacement deps, one version pin
# -- was reached by actually building and running this image and reading the real tracebacks, not
# guessed in advance; confirmed clean afterward with `pip show opencv-python-headless` (absent)
# and `python -c "import albumentations"` (succeeds) in the built image.
RUN pip install --no-cache-dir --no-deps "albumentations==1.3.1" qudida && \
    pip install --no-cache-dir "scikit-image>=0.16.1" "scikit-learn>=0.19.1" PyYAML

# RoIAlign.pytorch, per INSTALL.md ("Install third party modules"). Must be initialized on the
# host first -- see build_container.sh's own header; an uninitialized submodule COPYs as an empty
# directory and this build breaks with a missing setup.py.
#
# TORCH_CUDA_ARCH_LIST is required here: RoIAlign's setup.py builds its CUDA extension via
# torch.utils.cpp_extension, which -- absent this env var -- tries to auto-detect the target arch
# from a live GPU via `torch.cuda.get_device_capability()`, unavailable during `docker build` (no
# `--gpus` there, unlike `docker run`), and would otherwise either fail outright or silently fall
# back to a default arch list that doesn't include this repo's real GPUs (2x RTX 6000 Ada
# Generation, compute 8.9).
#
# Set to "8.6+PTX", NOT the literal "8.9" a first version of this line used -- a REAL build with
# "8.9" failed differently from every other real failure in this file: not missing/wrong software,
# but torch==1.8.1's OWN `torch.utils.cpp_extension._get_cuda_arch_flags()` hardcodes a table of
# archs it recognizes (2021-era, pre-dating Ada Lovelace entirely) and raises `ValueError: Unknown
# CUDA arch (8.9) or GPU not supported` for anything outside it -- UNLIKE the plain CMake
# `-DCMAKE_CUDA_ARCHITECTURES=89` this repo's baselines/pixel-perfect-sfm/Dockerfile and this same
# Dockerfile's own COLMAP build (above) use, which is just a number CMake passes straight to nvcc,
# with no such named-arch table to be out of date. "8.6" (Ampere) is the newest arch that table
# does recognize; "+PTX" additionally embeds forward-compatible PTX (not just Ampere SASS) that
# the CUDA driver JIT-compiles to compute_89 SASS on first run on the real Ada hardware -- the
# standard workaround for building an old torch extension for a GPU generation newer than that
# torch release's own compiled-in knowledge, and the reason this repo's real GPUs can still be
# targeted at all through this one torch-mediated build path, unlike the CMake-driven ones.
#
# --no-build-isolation: a REAL build of this file failed here without it -- pip's own default
# isolated build environment (PEP 517) is a fresh venv with only setuptools/wheel in it, and
# RoIAlign's setup.py (old-style, no pyproject.toml declaring torch as a build requirement) does a
# bare `import torch` at the top to reach torch.utils.cpp_extension -- real error: "Getting
# requirements to build wheel" ... "ModuleNotFoundError: No module named 'torch'". This is the
# standard fix for old torch C++/CUDA-extension packages built this way: skip the isolated build
# env entirely and build directly against this image's own venv, which already has torch==1.8.1
# installed (see above).
#
# sed patch to setup.py's own `if torch.cuda.is_available():` guard, unconditionally True: this
# is a SECOND, independent live-GPU check on top of the TORCH_CUDA_ARCH_LIST one above -- and this
# one isn't a target-arch question at all, it's whether the CUDAExtension (roi_align.
# crop_and_resize_gpu) gets added to `ext_modules` in the first place. `torch.cuda.is_available()`
# is unconditionally False during any `docker build` (no live GPU, ever, regardless of `--gpus`
# at `docker run` time later) -- so unpatched, this line silently built ONLY the CPU extension,
# every time, on every machine, and the built image looked fine (imports clean, no build error)
# right up until a REAL run reached `roi_align/crop_and_resize.py`'s OWN separate, and legitimate,
# runtime `if torch.cuda.is_available(): import roi_align.crop_and_resize_gpu` -- on an actual GPU
# machine, that one IS true, and the extension it tries to import was simply never built: real
# error "ModuleNotFoundError: No module named 'roi_align.crop_and_resize_gpu'", well past the
# timm fix above (through coarse SfM, into post-optimization's own model construction). This
# baseline's own sandbox testing during development couldn't catch it -- no GPU there either, so
# `crop_and_resize.py`'s runtime check also stayed False, silently skipping the very same import
# that only fails on hardware this Dockerfile is never built on. Only setup.py's copy is patched;
# crop_and_resize.py's own runtime check is left alone -- it's correct behavior for whatever
# machine actually runs the built image, GPU or not, and this sed only needs to fix the
# build-time question of whether the extension exists in the wheel at all.
COPY DetectorFreeSfM/third_party/RoIAlign.pytorch third_party/RoIAlign.pytorch
RUN sed -i 's/if torch\.cuda\.is_available():/if True:  # patched: see baselines\/DetectorFreeSfM\/Dockerfile/' \
        third_party/RoIAlign.pytorch/setup.py && \
    grep -q 'patched: see baselines/DetectorFreeSfM/Dockerfile' third_party/RoIAlign.pytorch/setup.py && \
    TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install --no-cache-dir --no-build-isolation \
        ./third_party/RoIAlign.pytorch

# baselines/DetectorFreeSfM/run_sequence.py's own needs, on top of DetectorFreeSfM's own
# requirements.txt: `trimesh` (reads back the dense fused.ply the same way
# baselines/colmap/run_sequence.py's own identical dense stage does), plus inspection_viz's own
# light (non-viser-heavy-USE, but still imported) runtime deps for the dataset loader + output
# writer -- same explicit subset baselines/depth_anything_3/Dockerfile and
# baselines/mpsfm/Dockerfile both install for the same reason (inspection_viz itself is not
# installed as a package; run_sequence.py imports it from the host checkout via run_container.sh's
# own PYTHONPATH). Not the top-level requirements.txt wholesale -- that file's own unpinned
# `pycolmap` would fight this image's pinned `pycolmap==0.3.0` above, depending on install order.
RUN pip install --no-cache-dir trimesh OpenEXR pillow viser pyyaml

# NOT prefetching the S2DNet weight here, deliberately, unlike baselines/pixel-perfect-sfm's own
# identical-URL fix for the same upstream weight file: that fix exists because pixsfm's own tree
# is COPY'd into the image and built in place, so it's root-owned at runtime, and the container's
# host-uid user can't write into it. Here, src/MultiviewMatcher/backbone/S2DNet/checkpoints/ is
# never baked into this image at all -- it's read from the host-mounted, host-owned
# DetectorFreeSfM/ checkout at runtime (see this file's own header), and run_container.sh runs the
# container as that same host uid/gid, so s2dnet.py's own lazy `wget` on first use writes into a
# directory the container user already owns -- no prefetch needed. Only relevant if the coarse
# matcher in use actually routes through S2DNet at all (the demo config's default,
# `NEUSFM_coarse_matcher: loftr_official`, does not).

# Same shared convention as every other baseline Dockerfile in this repo.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:$PATH' \
        'PS1="[detectorfreesfm-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT -- same as every other baseline here: get a shell (run_container.sh), then run
# `python eval_dataset.py +demo=dfsfm.yaml` (or another hydra_configs/ target) from
# DetectorFreeSfM's own repo root, per its own README.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
