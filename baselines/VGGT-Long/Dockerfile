# VGGT-Long baseline: environment for DengKaiCQ/VGGT-Long (chunk-based, loop-closed VGGT for
# kilometer-scale sequences), built by replaying the `VGGT-Long` submodule's own README setup
# steps (its "Environment Setup" section) as image layers instead of running them live.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment (python + pinned torch/cu118 + requirements.txt + the
# README's recommended-but-optional C++ Sim3 loop-closure solver). No repo code is baked in
# beyond what that install needs, and nothing lives at /workspace, so runner code is always
# whatever is on the host — edits never need a rebuild. Weights are NOT downloaded at build time
# — see the note near the bottom.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Unlike the other baselines' Dockerfiles, no deadsnakes PPA is needed: Ubuntu 22.04 already ships
# python3.10 as its default python3, matching the README's own `conda create -n vggt-long
# python=3.10.18`. python3.10-dev + build-essential are for setup.py's CUDAExtension (Python.h to
# compile, g++ to link); this is the "devel" (not "runtime") CUDA base image specifically so that
# extension's build finds a full toolkit — nvcc isn't actually needed (fastloop/solve.cpp is an
# Eigen-only sparse solve, no .cu sources) but the *unversioned* libcudart.so `-lcudart` links
# against only ships in "devel", not "runtime". libgl1/libglib2.0-0/... are opencv-python's runtime
# deps (README's problem 4a); curl/wget are for scripts/download_weights.sh (not run at build
# time, see below, but needed once you run it yourself in the container shell).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev git build-essential curl wget \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libopenexr-dev openexr \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv

# wheel isn't bundled into a fresh venv the way pip/setuptools are -- every --no-build-isolation
# source build below (xformers, sim3solve) skips pip's usual "auto-install build deps into a
# throwaway env" step, so without this here they fail with `error: invalid command 'bdist_wheel'`.
RUN pip install --no-cache-dir -U pip setuptools wheel

WORKDIR /opt/VGGT-Long

# Step 1 (README): pinned torch first ("Verified to work with CUDA 11.8 and torch 2.5.1"), before
# the rest of the submodule is even copied in, so unrelated repo edits don't bust this slow layer.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118

COPY VGGT-Long/ .

# requirements.txt pins `xformers == 0.0.28.post3`, but PyPI's xformers wheels only cover specific
# prebuilt torch+CUDA+python ABI combos -- for anything else pip silently falls back to a wheel
# missing its compiled CUDA kernels (memory-efficient attention etc. then just aren't available,
# no error). Build it from source instead, against the venv's own torch 2.5.1+cu118 and this
# "devel" image's matching nvcc, so it's actually compiled for what's here. FORCE_CUDA=1 makes
# setup.py build the CUDA extensions even though no GPU is attached at build time (same reason
# baselines/VGGT-SLAM and this file's own sim3solve build don't need one either); the arch list
# covers V100/T4/A100/A6000+3090/H100 -- trim it if you only need one. PyPI's sdist doesn't vendor
# xformers' cutlass/flash-attention git submodules, so this clones instead of `pip install
# --no-binary`. Installed *before* requirements.txt so that install finds this exact pinned
# version already satisfied and leaves it alone, instead of overwriting it with a wheel.
ENV FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
RUN git clone --depth 1 --branch v0.0.28.post3 --recursive https://github.com/facebookresearch/xformers.git /tmp/xformers \
    && pip install --no-cache-dir --no-build-isolation -v /tmp/xformers \
    && rm -rf /tmp/xformers

# requirements.txt's `faiss-gpu` has no installable wheels anymore -- this is the README's own
# documented fix (Possible Problems, 4b): swap it for the CUDA-versioned package, cu11 to match
# the cu118 torch above. Done here instead of editing the submodule's own requirements.txt.
#
# requirements.txt's own `xformers == 0.0.28.post3` line is dropped entirely (not just left to
# pip's "already satisfied" check), because it doesn't reliably no-op: xformers' setup.py stamps
# its version from `git describe` (e.g. "0.0.28.post3+<hash>" from the source build above, not the
# plain "0.0.28.post3" pip parses out of this file), so pip's resolver sees that as *not*
# satisfying the exact pin and reinstalls xformers from PyPI here -- silently overwriting the
# CUDA-enabled build with a mismatched, kernel-less wheel. Removing the line outright is what
# actually stops that, regardless of whatever the installed version string happens to be.
#
# pypose/matplotlib -- hard, top-level imports (vggt_long.py itself; loop_utils/sim3loop.py +
# loop_refinement.py) requirements.txt is missing -- are installed in the *same* pip invocation as
# requirements.txt, not a second one after it: requirements.txt pins numpy==1.26.1 (numba needs
# <=2.2, so this matters), but a pin only constrains the resolver within the command that names
# it -- a separate `pip install pypose matplotlib` afterward has no idea numpy was ever pinned,
# and pypose pulls in a numpy with no such ceiling, silently upgrading it ("Numba needs NumPy 2.2
# or less. Got NumPy 2.4"). One combined command resolves everything (including the pin) together.
RUN sed -i -e 's/^faiss-gpu$/faiss-gpu-cu11/' -e '/^xformers/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt pypose matplotlib
# Step 3 (README, recommended but optional -- a pure-Python Sim3 solver is used otherwise): the
# C++ loop-closure solver. `python setup.py install` is the README's own command, not
# `pip install -e .` -- it builds directly against this environment with no pip build-isolation
# sandbox to fight, so it only needs torch already importable here, which it is.
RUN python setup.py install

# Steps 2, 4 and 5 are intentionally skipped:
#  - Step 2, weights (~6 GiB across VGGT/Pi3/map-anything/SALAD/DINO/DBoW): not fetched at build
#    time, consistent with every other baseline here (they all fetch weights lazily on first run
#    instead) -- baking multi-GB downloads into the image would make the build slow and
#    network-flaky, and facebook/map-anything is a gated HF repo needing `huggingface-cli login`
#    anyway. Run it yourself once, inside the container shell (run_container.sh mounts your
#    $HF_HOME, so it only has to happen once):
#      ./scripts/download_weights.sh
#  - Step 4, the DBoW2 CPU-only VPR alternative, and step 5, mapanything (not vendored in this
#    submodule) are both explicitly optional in the README; add them here if you need them.

# Install inspection_dataset_tools requirements
COPY requirements.txt ./requirements_idt.txt
# Same class of bug as the pypose/matplotlib fix above, one command further down the chain:
# requirements_idt.txt's own `numpy` line is unpinned (by design -- it's shared across every
# baseline here, most of which don't care), so *this* separate pip invocation has no memory of
# VGGT-Long's own numpy==1.26.1 pin either, and one of scipy/opencv-python-headless/viser's own
# floor requirements pulls numpy up again (observed: to 2.2.6) -- which doesn't fail this install,
# but silently breaks numba's already-installed compiled extension (built against numpy 1.x's
# C-API), surfacing later as `ImportError: numpy.core.multiarray failed to import` the first time
# something actually imports numba (e.g. running run_sequence.py). Re-asserting the pin here too
# keeps this command's resolution consistent with the numba install two steps up, instead of
# editing the shared requirements.txt (which every other baseline's own numpy needs stay unpinned).
RUN pip install --no-cache-dir -r ./requirements_idt.txt numpy==1.26.1

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[vggt-long-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python vggt_long.py --image_dir <path> [--config configs/base_config.yaml]` yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
