# VGGT-SLAM baseline: environment for MIT-SPARK/VGGT-SLAM (dense feed-forward SLAM optimized on
# the SL(4) manifold), built by replaying the `VGGT-SLAM` submodule's own setup.sh as image
# layers instead of running it live.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a
# shell; the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment (python + VGGT-SLAM's pip deps + the four third_party
# packages setup.sh clones and installs editable: salad, the MIT-SPARK VGGT fork, Perception
# Encoder, SAM 3). No repo code is baked in beyond what that install needs, and nothing lives at
# /workspace, so runner code is always whatever is on the host — edits never need a rebuild.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) —
# pull a real, current 3.11 from deadsnakes instead, matching the `conda create -n vggt-slam
# python=3.11` the upstream README calls for. Everything is then pip-installed into its own venv
# (not the system python) so it can't collide with Debian-managed dist-packages (add-apt-
# repository itself pulls in distutils-installed python3-* packages).
#
# build-essential is here (unlike the other baselines' Dockerfiles) because salad/VGGT fork/PE/
# SAM3 are research-repo installs, not published wheels — some carry small C/Cython extensions
# in their setup.py. libsm6/libxext6/libxrender1 are for `opencv-python` (requirements.txt pulls
# the GUI-enabled build, not -headless), on top of the libgl1/libglib2.0-0 open3d and cv2 both
# need just to import.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv git build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get install -y libopenexr-dev openexr \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

RUN pip install --no-cache-dir -U pip

WORKDIR /opt/VGGT-SLAM

# setup.sh step 1: base requirements (torch/torchvision pinned here, so this is the slow,
# rarely-changing layer) — copied in on its own first so later repo edits don't bust it.
COPY VGGT-SLAM/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install OpenEXR

# setup.sh steps 2-5: clone and install (editable) VGGT-SLAM's four third_party packages, at the
# same URLs/paths setup.sh itself uses. Cloned fresh at build time rather than copied from the
# host: the submodule checkout doesn't vendor them (setup.sh creates third_party/ itself, and
# it's .gitignore'd), so a git clone here is the only way to reproduce what setup.sh would do.
RUN mkdir -p third_party \
    && git clone https://github.com/Dominic101/salad.git third_party/salad \
    && pip install --no-cache-dir -e ./third_party/salad \
    && git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt \
    && pip install --no-cache-dir -e ./third_party/vggt \
    && git clone https://github.com/facebookresearch/perception_models.git third_party/perception_models \
    && pip install --no-cache-dir -e ./third_party/perception_models \
    && git clone https://github.com/facebookresearch/sam3.git third_party/sam3 \
    && pip install --no-cache-dir -e ./third_party/sam3

# setup.sh step 6: the repo itself, editable. Copied in last (small, frequently-changing layer)
# so edits to vggt_slam/evals/main.py don't invalidate the slow layers above. This only overlays
# what the submodule checkout actually contains (no third_party/), so the clones above survive.
COPY VGGT-SLAM/ .
RUN pip install --no-cache-dir -e .

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[vggt-slam-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python main.py --image_folder <path> ...` (or main_realtime.py) yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
