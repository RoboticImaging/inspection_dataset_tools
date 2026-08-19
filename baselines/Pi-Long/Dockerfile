# Pi-Long baseline: environment for DengKaiCQ/Pi-Long (VGGT-Long's chunk/loop-closure framework
# migrated onto Pi3/Pi3X, for kilometer-scale sequences), built by replaying the `Pi-Long`
# submodule's own README setup steps (its "Environment Setup" section) as image layers instead of
# running them live.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment (python + pinned torch/cu118 + requirements.txt + the
# README's recommended-but-optional C++/CUDA Sim3 loop-closure solver). No repo code is baked in
# beyond what that install needs, and nothing lives at /workspace, so runner code is always
# whatever is on the host — edits never need a rebuild. Weights are NOT downloaded at build time
# — see the note near the bottom.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Unlike the Pi3/VGGT-SLAM/VGGT Dockerfiles, no deadsnakes PPA is needed: Ubuntu 22.04 already
# ships python3.10 as its default python3, matching the README's own `conda create -n pi-long
# python=3.10.18`. python3.10-dev + build-essential are for setup.py's CUDAExtension (Python.h to
# compile, g++ to link) and the from-source xformers build below; this is the "devel" (not
# "runtime") CUDA base image specifically so those finds a full toolkit — nvcc isn't actually
# needed by setup.py's own fastloop/solve.cpp (an Eigen-only sparse solve, no .cu sources) but the
# *unversioned* libcudart.so `-lcudart` links against only ships in "devel", not "runtime". No
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1 here (unlike the other Dockerfiles): Pi-Long's
# own requirements.txt doesn't pull opencv-python itself at all -- pi_long.py's `import cv2` is
# satisfied by inspection_dataset_tools' own opencv-python-headless (installed below), which needs
# none of those X11/GL libs. libopenexr-dev/openexr are still needed to build the `OpenEXR` python
# package that same shared requirements.txt pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev git build-essential curl wget \
        libopenexr-dev openexr \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv

# wheel isn't bundled into a fresh venv the way pip/setuptools are -- the --no-build-isolation
# xformers source build below skips pip's usual "auto-install build deps into a throwaway env"
# step, so without this here it fails with `error: invalid command 'bdist_wheel'`.
RUN pip install --no-cache-dir -U pip setuptools wheel

WORKDIR /opt/Pi-Long

# Step 1 (README): pinned torch first ("verified to work with torch 2.5.1", cu118), before the
# rest of the submodule is even copied in, so unrelated repo edits don't bust this slow layer.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118

COPY Pi-Long/ .

# Unlike VGGT-Long's requirements.txt, Pi-Long's own leaves `xformers` unpinned -- letting pip
# resolve it in the same command as everything else below would be free to pull in whatever
# xformers is newest on PyPI, which in turn can demand a newer torch than the 2.5.1+cu118 pinned
# above and silently upgrade it. Build a known-good version (the one VGGT-Long itself pins, and
# the same combo already verified against this exact torch/CUDA pair) from source instead, so it
# actually compiles its CUDA kernels against what's here rather than falling back to a
# kernel-less generic wheel. FORCE_CUDA=1 makes setup.py build the CUDA extensions even though no
# GPU is attached at build time (same reason this file's own sim3solve build below doesn't need
# one either); the arch list covers V100/T4/A100/A6000+3090/H100 -- trim it if you only need one.
# PyPI's sdist doesn't vendor xformers' cutlass/flash-attention git submodules, so this clones
# instead of `pip install --no-binary`.
ENV FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
RUN git clone --depth 1 --branch v0.0.28.post3 --recursive https://github.com/facebookresearch/xformers.git /tmp/xformers \
    && pip install --no-cache-dir --no-build-isolation -v /tmp/xformers \
    && rm -rf /tmp/xformers

# Drop requirements.txt's own unpinned `xformers` line entirely (not just leave it to pip's
# "already satisfied" check) before installing the rest: xformers' setup.py stamps its version
# from `git describe` (e.g. "0.0.28.post3+<hash>", not the plain "0.0.28.post3" pip parses out of
# a version pin), so an unpinned line would still make pip's resolver want to reinstall it from
# PyPI here -- silently overwriting the CUDA-enabled build above with a mismatched, kernel-less
# wheel. Everything else in requirements.txt (matplotlib, pypose, faiss-gpu-cu11, ...) installs
# as-is -- unlike VGGT-Long's, this submodule's requirements.txt already has those fixed up.
RUN sed -i '/^xformers$/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# requirements.txt is missing `plyfile` (confirmed by a real ModuleNotFoundError at runtime, not
# just a static read) -- pi3/utils/basic.py imports PlyData/PlyElement from it unconditionally at
# module load, so it's needed even though nothing in the default config path writes a .ply through
# that particular module.
RUN pip install --no-cache-dir plyfile

# Step 3 (README, recommended but optional -- a pure-Python Sim3 solver is used otherwise): the
# C++/CUDA loop-closure solver. `python setup.py install` is the README's own command, not
# `pip install -e .` -- it builds directly against this environment with no pip build-isolation
# sandbox to fight, so it only needs torch already importable here, which it is. Its Eigen headers
# (thirdparty/eigen-3.4.0) are vendored in the submodule itself, so no extra apt/clone step here.
RUN python setup.py install

# Steps 2 and 4 are intentionally skipped:
#  - Step 2, weights (Pi3/Pi3X + SALAD/DINO for loop closure, ~4.5-6 GiB depending on model_type):
#    not fetched at build time, consistent with every other baseline here (they all fetch weights
#    lazily on first run instead) -- baking multi-GB downloads into the image would make the build
#    slow and network-flaky. Run it yourself once, inside the container shell (run_container.sh
#    mounts your $HF_HOME, so it only has to happen once):
#      ./scripts/download_weights.sh
#  - Step 4, the DBoW CPU-only VPR alternative (its DPRetrieval C++ extension), is explicitly
#    optional in the README and off by default (Model.useDBoW: False in every shipped config);
#    add it here if you need it.

# Install inspection_dataset_tools requirements
COPY requirements.txt ./requirements_idt.txt
# Same class of bug as the xformers fix above, one command further down the chain: this shared,
# unpinned requirements.txt (by design -- most baselines don't care about numpy's version) has no
# memory of Pi-Long's own numpy==1.26.1 pin (needed by numba==0.61.2 above: "Numba needs NumPy 2.2
# or less"), and opencv-python-headless/viser's own floor requirements pull numpy up again if not
# re-asserted here -- which wouldn't fail this install, but silently breaks numba's
# already-installed compiled extension (built against numpy 1.x's C-API), surfacing later as
# `ImportError: numpy.core.multiarray failed to import` the first time something actually imports
# numba (e.g. running run_sequence.py).
RUN pip install --no-cache-dir -r ./requirements_idt.txt numpy==1.26.1

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[pi-long-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python pi_long.py --image_dir <path> [--config configs/base_config.yaml]` yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
