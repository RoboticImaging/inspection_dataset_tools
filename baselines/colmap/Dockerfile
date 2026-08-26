# syntax=docker/dockerfile:1
#
# COLMAP baseline: environment for colmap/colmap (the structure-from-motion / multi-view-stereo
# pipeline used elsewhere in this repo as a classical SfM reference/pose source). Built by
# replaying the `colmap` submodule's own `docker/Dockerfile` build recipe (see the submodule's
# `docker/README.md`, "Build from Scratch") as an image layer here, rather than pulling the
# official `colmap/colmap:latest` image (which `colmap/docker/run.sh` falls back to) -- this way
# the compiled binary stays version-matched to whatever commit the `colmap` submodule is pinned
# at, instead of drifting from it.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# Unlike a plain CLI wrapper, this image also carries a Python venv (see the runtime stage below)
# so baselines/colmap/run_sequence.py can run in it: run_sequence.py drives the classical
# COLMAP CLI as a subprocess (sparse SfM -> merge -> dense MVS, see its own module docstring),
# then reads the result back via pycolmap to write depth/pose/cloud predictions the same way
# every other baseline here does. `colmap` itself is still also usable directly as a CLI, the
# same way the submodule's own docker/run.sh uses it, e.g.:
#   colmap automatic_reconstructor --image_path ./images --workspace_path .
# or as individual pipeline steps (feature_extractor, exhaustive_matcher/sequential_matcher,
# mapper, ...) -- see `colmap --help` / colmap/doc/ for the full command reference.
#
# This is a two-stage build (builder -> runtime), exactly mirroring the submodule's own Dockerfile:
# the builder stage compiles COLMAP against the full CUDA "devel" image with every build
# dependency installed, then only the installed binaries/libraries (`/colmap-install`) are copied
# into a slim CUDA "runtime" image, so the final image doesn't carry the ~GB+ of compilers/headers
# needed only to build it.
#
# CUDA_ARCHITECTURES defaults to "all-major" per the submodule's own Dockerfile, which asks nvcc
# to target every major architecture it knows about. Override at build time
# (--build-arg CUDA_ARCHITECTURES=75) if you know your GPU's compute capability and want a faster
# build / smaller binary -- see the submodule's own docker/README.md, "Build from Scratch".

ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

#
# Builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ARG FETCHCONTENT_FULLY_DISCONNECTED=OFF
ENV CCACHE_DIR=/colmap/build/.ccache
ENV CCACHE_BASEDIR=/colmap

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building. Same package list as colmap/docker/Dockerfile.
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
        libmkl-full-dev

# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
RUN mkdir -p /usr/include/opencv4

# Copy the colmap submodule into the image. Unlike colmap/docker/Dockerfile (whose own build
# context is the submodule root, so it can `COPY . /colmap`), our build context is the repo root,
# so the submodule path is named explicitly.
COPY colmap/ /colmap

# Build and install COLMAP.
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} \
        -DBLA_VENDOR=Intel10_64lp && \
    ninja install

#
# Runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run the COLMAP binary compiled in the builder stage. Same list as
# colmap/docker/Dockerfile -- keeping the final image free of the build-only dependencies above.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libomp5 \
        libopengl0 \
        libmetis5 \
        libceres4t64 \
        libopenimageio2.4t64 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt6core6 \
        libqt6gui6 \
        libqt6widgets6 \
        libqt6openglwidgets6 \
        libqt6svg6 \
        libcurl4 \
        libssl3t64 \
        libmkl-locale \
        libmkl-intel-lp64 \
        libmkl-intel-thread \
        libmkl-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files from /colmap-install/ in the builder stage to /usr/local/ in the runtime stage
# (colmap/docker/Dockerfile's own layout) -- this puts the `colmap` binary on PATH and makes the
# image usable as a base for anything that compiles/links against COLMAP as a library.
COPY --from=builder /colmap-install/ /usr/local/

# Python env for run_sequence.py. Ubuntu 24.04 (this image's own base) already ships a current
# python3 (3.12) by default -- unlike baselines/vggt or baselines/Pi3's Ubuntu 22.04 base, no
# deadsnakes PPA is needed here. python3-venv bundles ensurepip, so no separate python3-pip
# package is needed either. libgl1/libglib2.0-0 are what opencv-python-headless needs to import
# (this repo's requirements.txt pulls the headless build, not the GUI one -- see
# baselines/Pi3/Dockerfile's own comment on why the GUI build needs more than this).
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir -U pip

WORKDIR /opt/colmap-env

# This repo's own requirements.txt, in full -- including pycolmap, which run_sequence.py uses to
# read the reconstruction back (poses, intrinsics) after driving the `colmap` CLI itself as a
# subprocess. This is the plain prebuilt PyPI `pycolmap` wheel: a separate process from the
# from-source-compiled `colmap` CLI binary above (invoked over subprocess, never linked), so it
# carries none of baselines/mpsfm/Dockerfile's documented pyceres/pycolmap from-source ABI
# fragility -- that risk is specific to building pyceres *and* pycolmap *and* a patched colmap
# fork from source together against one shared pybind11 version; this is just a normal wheel
# install of an independent binary.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterward: inside the container,
# `python` must always be /opt/venv/bin/python. `colmap` itself stays reachable regardless, at
# /usr/local/bin (already on PATH from the base image, unaffected by the venv prepend above).
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:$PATH' \
        'PS1="[colmap-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python run_sequence.py <config...>` yourself, or `colmap <command> ...` directly (see
# colmap --help, or colmap/doc/ for the full pipeline).
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
