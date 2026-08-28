# syntax=docker/dockerfile:1
#
# VidMap baseline: environment for cvg/vidmap ("VidMap: Exploiting Temporal Structure for
# Video-Based Structure-from-Motion", ECCV 2026) -- extends COLMAP's global mapping pipeline with
# temporal tracks, loop closures (MegaLoc retrieval), and metric depth priors (Depth-Anything-3),
# using RoMaV2 for dense matching. Its own README ("Setup") gives three steps and no Docker
# recipe of its own:
#   1. build and install COLMAP 4.1 + PyCOLMAP bindings from source
#   2. pip install PyTorch, TorchVision, xFormers (CUDA-specific, not in pyproject.toml's own
#      pinned dependency list -- a manual step by design, like every other baseline here that
#      needs a matching CUDA build)
#   3. pip install -e .   (builds vidmap's own C++/pybind11 extension, `vidmap_native`, via
#      `find_package(colmap CONFIG REQUIRED)` against the COLMAP installed in step 1 -- see
#      vidmap/extensions/colmap/CMakeLists.txt:29)
# This Dockerfile follows that order, with step 1 reusing baselines/colmap/Dockerfile's own
# from-source build recipe (same apt package list, same cmake/ninja invocation) -- unlike
# baselines/pixel-perfect-sfm, which needed a completely separate, much older COLMAP 3.8 checkout
# after real build failures (see that Dockerfile's own header for the full four-risk story).
# VidMap's own version target is COLMAP **4.1.0** (vidmap/extensions/colmap/CMakeLists.txt:22-28,
# `VIDMAP_COLMAP_VERSION`/`VIDMAP_COLMAP_REVISION`, the latter confirmed to be exactly colmap's
# own `4.1.0` git tag) -- much closer to this repo's own `colmap` submodule (`4.2.0.dev0`, 114
# commits past the merge-base with that tag) than pixel-perfect-sfm's multi-major-release gap
# against COLMAP 3.8 ever was, and vidmap's own CMakeLists.txt only *warns*, not fatal-errors, on
# a version mismatch (lines 30-36) -- so reusing this repo's own submodule, the way `colmap`/
# `glomap` already do, is the right call here rather than pinning a second separate checkout.
#
# Build: build_container.sh (build context is the repo root, so both submodules are copied in).
#
# Like pixel-perfect-sfm, this is a SINGLE stage, not builder+runtime: vidmap_native's own compiled
# extension (and the many Python packages layered on top of it -- torch, xformers, RoMaV2,
# Depth-Anything-3, MegaLoc) are simpler to keep colocated with the full build toolchain than to
# curate a slim runtime for.
#
# Two real build failures already found and fixed here, from actual build output (not
# pre-empted by guessing):
#   - vidmap_native declares no find_package(Eigen3/Boost/Ceres/...) of its own at all
#     (vidmap/extensions/colmap/CMakeLists.txt) -- it gets all of those transitively through
#     COLMAP's own exported CMake targets (colmap::colmap, colmap::colmap_estimators). But it
#     DOES manually link `absl::log absl::check` (lines 76-79), because "COLMAP's static
#     libraries can reference Abseil's logging internals without listing Abseil in their exported
#     configuration". `libabsl-dev` alone does NOT fix this: real error seen --
#     "CMake Error ... target_link_libraries: Target vidmap_native_algorithms links to:
#     absl::log; but the target was not found" -- even with it installed. Inspecting the actual
#     .deb confirmed why: Ubuntu 24.04's own `libabsl-dev` is Abseil 20220623 (June 2022), which
#     genuinely predates absl::log/absl::check entirely (find_package(absl CONFIG) does succeed --
#     a real abslConfig.cmake ships in the package -- there is just nothing log/check-related
#     anywhere inside it to find). Fixed below by building a current Abseil LTS release from
#     source instead, installed to /usr/local before COLMAP is even built.
#   - PoseLib (one of vidmap's own pinned git+ dependencies) generates pybind11 stubs via a CMake
#     custom command that calls `${Python_EXECUTABLE} .../generate_stubs.py` -- i.e. this venv's
#     own interpreter, directly, as a raw subprocess -- even though PoseLib's own pyproject.toml
#     `[build-system] requires` correctly lists `pybind11-stubgen==2.5.1` as a *build* dependency:
#     that pin only lands inside pip's isolated, throwaway build environment, which the CMake
#     custom command never sees. Real error seen: "/opt/venv/bin/python3: No module named
#     pybind11_stubgen" followed by the whole `pip install -e .` failing (poselib's build treats
#     a failed stub-generation step as a hard build failure, not a soft warning). Fixed below by
#     pre-installing that exact same version into the venv itself, before `pip install -e .` ever
#     runs, so the outer interpreter PoseLib's own CMake step actually invokes already has it.
#
# Everything past these two hasn't been build-tested yet -- treat what follows as still
# provisional in the same way every other baseline's first Dockerfile attempt here was:
#   - vidmap/extensions/colmap/src/stages/depth_prior.h includes
#     colmap/estimators/cost_functions/utils.h and uses colmap::EigenQuaternionMap<T>/
#     EigenVector3Map<T> directly -- on this repo's own colmap submodule those aliases moved to a
#     NEW header (quaternion_utils.h, added by upstream commit 1ee277a3) and no longer live in
#     utils.h. This currently compiles only by transitive-include luck (bundle_adjustment.cc
#     includes reprojection_error.h first, which now pulls in the new header before depth_prior.h
#     is reached) -- a real but currently-dormant fragility, not pre-emptively patched here.
#   - GPU-accelerated mapping (global positioning + bundle adjustment) needs Ceres 2.3+ built with
#     CUDA and cuDSS support (README, "GPU acceleration") -- deliberately NOT attempted here,
#     matching this repo's practice of not pre-solving speculative extra complexity: plain apt
#     `libceres-dev` (CPU-only mapper) is the default, same as baselines/colmap/baselines/glomap.
#   - xformers==0.0.31 is installed from the same cu128 wheel index as torch/torchvision below, as
#     a prebuilt-wheel attempt -- if no matching wheel exists for this exact
#     torch==2.7.1+python3.12+cu128 combination, fall back to building it from source the way
#     baselines/VGGT-Long/Dockerfile already does for an older xformers pin (clone, `pip install
#     --no-build-isolation`, FORCE_CUDA=1).
#   - The first `vidmap.frontend` run downloads ~9 GB of model checkpoints (README) -- not
#     attempted at build time, matching every other baseline's lazy-weight-fetching convention.
#
# The `third_party/RoMaV2`, `third_party/Depth-Anything-3`, and `third_party/MegaLoc` nested
# submodules inside vidmap (its own .gitmodules) must already be checked out on the host -- all
# three are mandatory, non-optional defaults in vidmap's own FrontendConfig, not opt-in extras --
# run `git -C vidmap submodule update --init --recursive` before building, or `COPY vidmap/
# /vidmap` below will copy empty directories.

ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ARG FETCHCONTENT_FULLY_DISCONNECTED=OFF
ENV CCACHE_DIR=/colmap/build/.ccache
ENV CCACHE_BASEDIR=/colmap
ENV DEBIAN_FRONTEND=noninteractive

# Step 1's build dependencies: baselines/colmap/Dockerfile's own builder-stage package list
# verbatim (NOT libabsl-dev -- see this file's own header for why that apt package can't satisfy
# vidmap_native's own absl::log/absl::check link requirement, fixed below via a from-source
# build instead), plus git/python3/python3-venv/python3-dev (needed by steps 2-3's `pip install`,
# not by colmap itself), and the apt packages opencv-python's GUI-enabled wheel needs to import
# (vidmap/pyproject.toml pins opencv-python==4.11.0.86, not -headless -- same list already
# established and documented by baselines/vggt/Dockerfile and baselines/Pi3/Dockerfile for the
# same wheel).
RUN apt-get update && \
    apt-get install -y \
        ccache \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libopenimageio-dev \
        openimageio-tools \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qt6-base-dev \
        libqt6opengl6-dev \
        libqt6openglwidgets6 \
        qt6-svg-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libmkl-full-dev \
        git \
        python3 \
        python3-venv \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1

# Fix issue in Ubuntu's openimageio CMake config (same as baselines/colmap/Dockerfile).
RUN mkdir -p /usr/include/opencv4

# Abseil, from source -- see this file's own header for why Ubuntu's own libabsl-dev (20220623,
# June 2022) can't satisfy vidmap_native's absl::log/absl::check link requirement at all. Built
# and installed to /usr/local before COLMAP (next step) so that if COLMAP's own build ever reaches
# for Abseil transitively (e.g. via glog), it resolves to this same copy rather than a second,
# potentially ABI-incompatible one. ABSL_ENABLE_INSTALL=ON: off by default, required for Abseil's
# own CMake to emit an installable/exported config at all -- without it this still builds fine but
# leaves nothing for find_package(absl CONFIG) to find, the same symptom as the apt package's own
# gap, just for a different reason. CMAKE_POSITION_INDEPENDENT_CODE=ON: Abseil's own CMake
# defaults to static libraries (Google's own guidance -- a shared Abseil is fragile across
# differing build flags/versions), but a static lib linked into vidmap_native's own shared .so
# extension needs to have been compiled with -fPIC in the first place, or the final link fails
# with relocation errors. ABSL_PROPAGATE_CXX_STD=ON is Abseil's own recommended flag so consumers
# pick up a consistent C++ standard rather than whatever CMake would otherwise default to.
RUN git clone --branch 20260817.0 --depth 1 https://github.com/abseil/abseil-cpp.git /abseil && \
    mkdir -p /abseil/build && \
    cd /abseil/build && \
    cmake .. \
        -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DABSL_ENABLE_INSTALL=ON \
        -DABSL_BUILD_TESTING=OFF \
        -DABSL_PROPAGATE_CXX_STD=ON && \
    ninja install

# Step 1: build and install COLMAP to /usr/local -- a normal system location, so
# `find_package(colmap CONFIG REQUIRED)` in step 3 below finds it with no extra
# CMAKE_PREFIX_PATH hint, the same way baselines/colmap/Dockerfile's runtime stage and
# baselines/pixel-perfect-sfm/Dockerfile both already rely on.
COPY colmap/ /colmap
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} \
        -DBLA_VENDOR=Intel10_64lp && \
    ninja install

# Python venv. Ubuntu 24.04's own python3 (3.12) is within torch 2.7.1's cu128 wheels' supported
# range (3.9-3.13), so no deadsnakes PPA is needed here, unlike baselines/depth_anything_3's
# older-torch Dockerfile.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -U pip setuptools wheel

WORKDIR /vidmap

# Its own third_party/RoMaV2, third_party/Depth-Anything-3, and third_party/MegaLoc submodules
# must already be checked out on the host (see this file's own header) -- COPY only mirrors
# whatever is actually on disk.
COPY vidmap/ .

# Step 2: pinned CUDA-specific torch/torchvision/xformers, before vidmap's own dependencies are
# resolved -- README's own tested combination ("PyTorch 2.7.1, TorchVision 0.22.1, xFormers
# 0.0.31"). cu128 (not an unpinned `pip install torch`, and not a newer cuXXX) matches this
# repo's own established reasoning in baselines/depth_anything_3/Dockerfile: it's within this
# host's driver-reported max CUDA (checked via `nvidia-smi` at the time this file was written),
# where an unpinned install can silently resolve a build needing a newer driver than most hosts
# have. See this file's own header for the xformers-from-source fallback if this wheel doesn't
# exist for this exact combination.
RUN pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir xformers==0.0.31 \
        --index-url https://download.pytorch.org/whl/cu128

# PoseLib (one of vidmap's own pinned git+ dependencies below) generates pybind11 stubs via a
# CMake custom command that calls this venv's own interpreter directly as a raw subprocess -- see
# this file's own header for why PoseLib's own `pybind11-stubgen==2.5.1` build-time pin doesn't
# reach that subprocess on its own (it only lands in pip's throwaway isolated build env). Matching
# that exact pinned version here, rather than an unpinned `pip install pybind11-stubgen`, since
# that's the version PoseLib's own CMakeLists.txt/generate_stubs.py was actually written against.
RUN pip install --no-cache-dir pybind11-stubgen==2.5.1

# Step 3: vidmap's own C++ extension + Python package, in editable mode (its own pyproject.toml
# drives scikit-build-core/cmake under the hood against the system colmap installed above), plus
# its pinned git+ dependencies (poselib, lightglue, geocalib) -- `git` is already installed above.
RUN pip install --no-cache-dir -e .

# This repo's own requirements.txt, last -- installed after vidmap's own exact-pinned
# dependencies so pip's resolver never has a reason to touch torch/xformers/numpy/opencv once
# they're already satisfied (same ordering rationale as every other baseline's Dockerfile here).
COPY requirements.txt /opt/requirements_idt.txt
RUN pip install --no-cache-dir -r /opt/requirements_idt.txt

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterward: inside the container,
# `python` must always be /opt/venv/bin/python. `colmap` itself stays reachable regardless, at
# /usr/local/bin (already on PATH from the base image, unaffected by the venv prepend above).
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:$PATH' \
        'PS1="[vidmap-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh, not yet
# written -- this Dockerfile and build_container.sh are the first step), where you then run
# `python -m vidmap.run ...` (or `.frontend`/`.map` separately) yourself, per vidmap/README.md.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
