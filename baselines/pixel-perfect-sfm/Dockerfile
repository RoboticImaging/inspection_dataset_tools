# syntax=docker/dockerfile:1
#
# Pixel-Perfect SfM baseline -- PRIMARY Dockerfile (Ubuntu 20.04 / CUDA 11.1.1, from-source Eigen
# 3.4.0, Ceres 2.1.0, COLMAP 3.8 with GUI_ENABLED at its own default). This replaced an earlier
# Ubuntu 22.04 / CUDA 11.8 build -- preserved at `Dockerfile.legacy-ubuntu22-cuda11.8.bak` in this
# directory, and as the `inspection-baselines/pixel-perfect-sfm-legacy` image tag -- which built
# and ran this baseline end-to-end but hit an unresolved, INTERMITTENT native crash (sometimes
# SIGSEGV, sometimes SIGABRT -- non-deterministic across runs) inside pixsfm's own C++ code,
# between COLMAP's own "Building matching graph..." log line and pixsfm's own "Start feature-metric
# keypoint adjustment." line -- i.e. inside `refiner.refine_keypoints_from_db(...)` in this
# directory's own run_sequence.py. An AddressSanitizer investigation against that build had stalled
# without a root cause when this file was tried instead.
#
# This file started as an INDEPENDENTLY-BUILT second environment, adapted from a real, publicly
# posted recipe, purely to see whether the same crash reproduced on a materially different
# toolchain -- it deliberately does not carry over most of the legacy file's own compatibility
# workarounds; see "Intentionally dropped" below for why each one turned out to be unnecessary
# here. It was promoted to primary after validation: 10/10 clean runs of the direct
# `refine_keypoints_from_db` API call, plus two full clean end-to-end `run_sequence.py` runs
# (feature extraction -> matching -> keypoint adjustment -> COLMAP mapper -> dense MVS on both
# GPUs -> stereo fusion -> pose export) against the same 99-image dataset the crash was originally
# found on -- no reproduction either way. Stated honestly: many things differ from the legacy image
# at once (Ubuntu 20.04 vs 22.04, CUDA 11.1 vs 11.8, GCC 9 vs 11/12, Eigen from source vs apt,
# GUI_ENABLED default-on vs forced off, Ceres build flags), so this doesn't prove which specific
# variable mattered -- only that this whole alternate stack doesn't reproduce the crash.
#
# Two bugs elsewhere in this repo surfaced (and were fixed) while validating this file, unrelated
# to the crash investigation itself: this image's Python 3.8 (vs. the legacy image's 3.10) doesn't
# support the PEP 604/585 syntax (`X | Y`, `dict[...]`) this repo's shared `inspection_viz`/
# `baselines.common` modules use, fixed via `from __future__ import annotations` in nine files
# (each with its own comment) plus a whole-class-vs-per-field `typing.get_type_hints` fix in
# `baselines/common/config.py`'s `_from_dict`; and this baseline's own `run_sequence.py` had one
# latent pycolmap API-version bug (`cam_from_world()` -> `projection_matrix()` for pycolmap 0.4.0,
# see that function's own docstring) that the legacy image's crash had always masked by never
# reaching that code path.
#
# Provenance: adapted from a Dockerfile posted at
# https://github.com/cvg/pixel-perfect-sfm/issues/41 (comment dated 2022-04-01, author Prajwal
# Chidananda), labelled "working Dockerfile" by its own author after a week of community
# troubleshooting on that issue (root cause of the bug that thread was chasing: Ubuntu's packaged
# `libeigen3-dev` was 3.3.7, one minor version below what Ceres actually needed, causing a linker
# error building pixsfm's own bindings -- fixed there by building Eigen 3.4.0 from source instead).
# Two much later comments on the same issue (2025-03-26 "MACILLAS", 2025-05-22 "GibsDev") record
# that by 2025 `git checkout dev` no longer worked at all, and both had already independently
# repinned COLMAP to tag `3.8` instead -- GibsDev specifically on an RTX 4090 (Ada, compute 8.9),
# needing `pycolmap==0.6.0` and `CMAKE_CUDA_ARCHITECTURES` including `89`. The pycolmap version
# bump isn't replicated here: this file still installs pixel-perfect-sfm's own `requirements.txt`
# unmodified (pycolmap==0.4.0, pyceres==1.0) -- but the `CMAKE_CUDA_ARCHITECTURES` need turned out
# to apply here too (see "CUDA architecture" below).
#
# The one real hypothesis under test: Eigen built explicitly from source (tag 3.4.0), rather than
# relying on the OS's own packaged Eigen, on the theory that some Eigen/Ceres/COLMAP/pixsfm ABI or
# version mismatch was behind the crash above. Stated honestly, though: Ubuntu 22.04's own
# `libeigen3-dev` (used by the legacy Dockerfile) is ALREADY 3.4.0 in practice -- so this file's
# Eigen version was not actually different from the legacy one's, just its *install mechanism*
# (source vs. apt package) and everything *around* it (Ubuntu 20.04 vs 22.04, an older
# CUDA/cuDNN/GCC, COLMAP built with GUI_ENABLED at its own default instead of forced OFF, Ceres
# built without forcing BUILD_SHARED_LIBS). The crash not reproducing doesn't prove
# "Eigen-from-source fixed it" -- only that this whole alternate stack doesn't reproduce it (see
# the promotion note above); isolating that one variable specifically would need a further,
# separate experiment.
#
# Compatibility risk carried over unavoidably -- COLMAP ref: the reference recipe's `git checkout
# dev` cannot be used verbatim. COLMAP's `dev` branch no longer exists upstream at all (confirmed
# directly against github.com/colmap/colmap). Falling back to `main` would reproduce the legacy
# Dockerfile's own documented "Compatibility risk #1" exactly: COLMAP's `src/base/` was reorganized
# to `src/colmap/sensor/` (`camera_models.h` -> `models.h`; confirmed both against this repo's own
# newer `colmap` submodule, at COLMAP_VERSION 4.2.0.dev0, and live against colmap/colmap's actual
# `main` branch), which breaks pyceres==1.0's hardcoded `#include colmap/base/camera_models.h` and
# pycolmap==0.4.0's CMake `${COLMAP_INCLUDE_DIRS}` Eigen-transitivity assumption -- both hard-pinned,
# unconditionally, by pixel-perfect-sfm's own requirements.txt, regardless of which Dockerfile
# builds COLMAP. So COLMAP is pinned to tag `3.8` below, exactly like the legacy Dockerfile -- not a
# stylistic choice, a hard requirement of the exact same pycolmap/pyceres pins this file also
# installs. One convenient consequence: this directory's own run_sequence.py already has its
# `_flags()` skip/rename logic written specifically against COLMAP 3.8's CLI, so pinning here to the
# same tag needs zero changes there.
#
# Intentionally dropped from the legacy Dockerfile (each hypothesized, not proven, unnecessary on
# this file's own different base image -- added back only if a real build/run error here
# demonstrates otherwise, same as every fix in that file was originally found; all confirmed still
# unneeded by the full validation run described above):
#   - The Kitware CMake 3.28.3 bootstrap: that existed solely because `-DCMAKE_CUDA_ARCHITECTURES=
#     all-major` needs CMake >= 3.23, and this file passes a plain numeric value instead (see
#     "CUDA architecture" below) -- so Ubuntu 20.04's own apt `cmake` (~3.16) is used as-is.
#   - The `-std=c++14` -> `-std=c++17` sed patch to COLMAP's src/CMakeLists.txt: that file's own
#     comment already noted this was harmless-but-not-required against Ceres 2.1.0 (which, unlike
#     2.2+, doesn't force C++17); nothing here changes that.
#   - The `sift.cc` GL_LUMINANCE sed patch: that fixed a `GUI_ENABLED=OFF` + `CUDA_ENABLED=ON`
#     combination upstream never exercised. This file leaves `GUI_ENABLED` at COLMAP's own default
#     (ON, since qtbase5-dev/libqt5opengl5-dev are installed below) precisely so that broken branch
#     is never hit -- COLMAP's CLI tools (feature_extractor, mapper, patch_match_stereo, ...) run
#     fine headless in this container (no DISPLAY/X11 socket) even built with GUI support, confirmed
#     across the full validation runs described above, not just assumed.
#   - `setuptools<82` (pkg_resources removal) and `numpy<2`+`opencv-python-headless==4.11.0.86`
#     (pybind11-v2.8-vs-numpy-2.x ABI break): both were needed against Ubuntu 22.04's Python 3.10.
#     Ubuntu 20.04's own default `python3` is 3.8.x, and NumPy 2.0 requires Python >= 3.9 -- so an
#     unconstrained `pip install numpy` here cannot resolve to numpy>=2 regardless of any pin,
#     confirmed by real build output (pip resolved numpy==1.24.4 unprompted). If either symptom
#     ever reappears, the fix is already known -- see the legacy Dockerfile.
#   - hloc (Hierarchical-Localization): the reference recipe clones and `pip install -e`s it, but
#     pixel-perfect-sfm's own requirements.txt has no hloc dependency, and it's only imported by
#     `pixsfm.refine_hloc`/`pixsfm.util.hloc` -- this directory's own run_sequence.py only ever uses
#     `pixsfm.refine_colmap.PixSfM`. Dropped to reduce build surface; trivial to add back (the same
#     two lines) if a future baseline here wants refine_hloc.
#   - Fetching pixel-perfect-sfm fresh via `git clone --recursive`: it's already a pinned submodule
#     of this repo, checked out on disk with its own nested third-party/{pybind11,HighFive}
#     submodules -- COPY'd in below instead, same as the legacy Dockerfile does.
#
# CUDA architecture -- this repo's actual GPUs are 2x RTX 6000 Ada Generation (compute 8.9 /
# sm_89). CUDA 11.1.1 (this file's base image) predates official sm_89 support entirely, so this
# was flagged as a real, immediate risk when this file was first tried -- not "robustness across
# many machines", but "does this base image even support this specific machine's hardware at all".
# Two things were then discovered from real build output, not assumed: (1) COLMAP 3.8's own
# CMakeLists.txt (line 255) hard-fails configure if `CMAKE_CUDA_ARCHITECTURES` isn't set at all --
# real error: "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc." --
# so it can't simply be left unset the way the reference recipe's own bare `cmake ..` does; and
# (2) CUDA 11.1.1's nvcc, once given `-DCMAKE_CUDA_ARCHITECTURES=89` explicitly, compiled COLMAP's
# CUDA code (patch_match_cuda.cu, SiftGPU/ProgramCU.cu, PBA/ProgramCU.cu) without error, and that
# compiled code ran dense MVS correctly on both real Ada Lovelace GPUs during validation --
# contradicting this file's own original assumption (and the legacy Dockerfile's own header) that
# CUDA >= 11.8 is required for compute 8.9. Hardcoded to `89` below for this specific machine, not
# a general multi-arch value or an ARG -- ARGs for base image version or
# `CMAKE_CUDA_ARCHITECTURES`, or auto-detecting the host's GPU, remain a deliberate follow-up (the
# "make it robust across a variety of GPU architectures" half of the user's own two-phase framing),
# not done here. Converting either to an ARG later is a two-line change, not a redesign.
#
# Build: same convention as every other baseline here -- build_container.sh, build context is the
# repo root:
#   baselines/pixel-perfect-sfm/build_container.sh
# The legacy Ubuntu 22.04/CUDA 11.8 image remains available, unpromoted, as
# `inspection-baselines/pixel-perfect-sfm-legacy` (built from `Dockerfile.legacy-ubuntu22-cuda11.8.bak`
# in this directory) if this build ever needs to be rolled back or compared against again.

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Reference recipe's own apt list, minus `python-dev`/`python-setuptools` (Python 2 -- nothing in
# the pycolmap/pyceres/pixsfm stack needs it; COLMAP 3.8's CMakeLists.txt looks for `python3`/
# `Python3`, never plain `python`), and minus `libeigen3-dev` (deliberately not installed at all --
# see the header above: the entire point here is Eigen from source, so nothing should compete with
# it on the include path). ccache/ninja-build also not added (reference recipe uses plain
# `make -j`; keeping that for Phase-1 fidelity -- a build-speed-only difference, not correctness).
# PLUS `libflann-dev` and `libsqlite3-dev`, both absent from the reference recipe but required:
# COLMAP 3.8's own cmake/FindFLANN.cmake and cmake/FindSQLite3.cmake hard-fail configure without
# them -- real errors seen, one at a time: "CMake Error at cmake/FindFLANN.cmake:91 (message):
# Could not find FLANN", then (after adding libflann-dev) "CMake Error at
# cmake/FindSQLite3.cmake:91 (message): Could not find SQLite3". (Dockerfile.legacy-ubuntu22-cuda11.8.bak
# already carries both packages for the same reason; the reference recipe's own COLMAP `dev`
# checkout apparently didn't need either, e.g. via different dependency-detection paths at that
# 2022 vintage.)
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        git \
        cmake \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-test-dev \
        libsuitesparse-dev \
        libfreeimage-dev \
        libflann-dev \
        libsqlite3-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libatlas-base-dev \
        libhdf5-dev \
        curl \
        python3-venv \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

# Eigen 3.4.0 from source -- see header. Header-only; a bare cmake/make install just installs the
# headers to /usr/local/include/eigen3, matching upstream Eigen's own install instructions. Must
# happen before Ceres/COLMAP below so their own find_package(Eigen3) resolves to this build (no
# libeigen3-dev is installed above to conflict with it).
WORKDIR /opt
RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
RUN cd eigen && mkdir build && cd build && cmake .. && make install

# Ceres 2.1.0 (final release, NOT the reference recipe's 2.1.0rc2 -- Dockerfile.legacy-ubuntu22-cuda11.8.bak
# already proved this exact final tag compatible with both pycolmap==0.4.0 and pyceres==1.0; a
# release candidate is a strictly worse choice here at zero extra cost). BUILD_TESTING/
# BUILD_EXAMPLES=OFF per the reference recipe, for build time only. BUILD_SHARED_LIBS left unset
# (reference recipe doesn't set it either, defaulting to a static build) -- if a link error later
# surfaces in pycolmap/pyceres/pixsfm against this, that's a real, evidence-based reason to revisit,
# not a preemptive guess.
WORKDIR /opt
RUN git clone --branch 2.1.0 --depth 1 https://github.com/ceres-solver/ceres-solver.git
WORKDIR /opt/ceres-solver
RUN mkdir build
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
RUN make -j
RUN make install

# COLMAP, pinned to tag 3.8 -- NOT `dev` (see header: that branch no longer exists upstream, and
# `main` reproduces Dockerfile.legacy-ubuntu22-cuda11.8.bak's own documented Compatibility risk #1
# regardless of base image, since pyceres==1.0/pycolmap==0.4.0 are unconditionally hard-pinned to
# COLMAP 3.8-era headers/CMake variables). GUI_ENABLED stays at COLMAP's own default (ON, given
# qtbase5-dev/libqt5opengl5-dev above) -- confirmed working headless across full validation, see
# header's "Intentionally dropped" section.
WORKDIR /opt
RUN git clone --branch 3.8 --depth 1 https://github.com/colmap/colmap.git
WORKDIR /opt/colmap
RUN mkdir build
WORKDIR /opt/colmap/build
# CMAKE_CUDA_ARCHITECTURES=89, NOT left unset or `all-major` -- see header's "CUDA architecture"
# section for the full story (COLMAP 3.8 requires this be set explicitly at all, and this base
# image's old apt cmake hits the legacy FindCUDA path where `all-major` isn't meaningful anyway;
# `89` is this machine's actual GPUs, RTX 6000 Ada Generation, confirmed both to compile and to run
# dense MVS correctly at runtime on real hardware, despite predating official sm_89 support).
RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
RUN make -j
RUN make install

# Python venv (repo convention, unlike the reference recipe's bare system pip3): keeps `python`/
# `pip` pointed at this image's own install regardless of what a host ~/.bashrc (bind-mounted in by
# run_container.sh) might otherwise inject onto PATH (e.g. a conda/mamba hook). No setuptools/numpy
# pins here -- see header's "Intentionally dropped" section for why both are expected to be
# structurally unnecessary on this base image's Python 3.8.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -U pip wheel

WORKDIR /pixel-perfect-sfm

# COPY, not `git clone --recursive` (reference recipe's own approach) -- pixel-perfect-sfm is
# already a pinned submodule of this repo, checked out on disk with its own nested
# third-party/{pybind11,HighFive} submodules already initialized (run
# `git -C pixel-perfect-sfm submodule update --init --recursive` on the host first if not).
COPY pixel-perfect-sfm/ .

# Builds pycolmap==0.4.0 and pyceres==1.0 from source (both `git+https://...@vX` pins in this
# file's own requirements.txt) against the COLMAP/Ceres just installed above.
RUN pip install --no-cache-dir -r requirements.txt

# pixsfm's own C++ extension + Python package, editable install (setup.py runs cmake/ninja
# internally). hloc intentionally not installed -- see header.
RUN pip install --no-cache-dir -e .

# S2DNet weight prefetch -- unrelated to which base image/toolchain this is; run_container.sh runs
# this container as the *host's* uid/gid (unaffected by any of the changes in this file), which has
# no write access to this root-owned COPY'd-in tree, so pixsfm's own lazy `wget` of these weights on
# first use would fail with a permission error. Same URL/target filename pixsfm's own
# pixsfm/features/models/s2dnet.py uses, so its own `if not path.exists()` check is satisfied and
# this download path never triggers again.
RUN mkdir -p pixsfm/features/models/checkpoints && \
    curl -fsSL https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth \
        -o pixsfm/features/models/checkpoints/s2dnet_weights.pth

# This repo's own requirements.txt, last -- same rationale as Dockerfile.legacy-ubuntu22-cuda11.8.bak:
# installed after pixsfm's own fragile from-source pins so pip's resolver has no reason to touch
# pyceres/pycolmap/torch once already satisfied.
COPY requirements.txt /opt/requirements_idt.txt
RUN pip install --no-cache-dir -r /opt/requirements_idt.txt

# Same shared convention as every other baseline Dockerfile in this repo.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:$PATH' \
        'PS1="[pixsfm-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT -- same as every other baseline here: get a shell (run_container.sh), then run
# `python run_sequence.py <config...>` or use `python -m pixsfm...`/`colmap ...` directly.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
