# DA3-Streaming baseline: environment for depth-anything-3's own `da3_streaming/` chunked,
# loop-closed streaming pipeline (built on Depth Anything 3 + ideas from VGGT-Long -- see
# depth-anything-3/da3_streaming/README.md), on top of the same base ../depth_anything_3/Dockerfile
# already builds (torch/xformers, depth-anything-3 installed editable at /opt) plus da3_streaming's
# own extra requirements.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment. No repo code is baked in beyond what depth-anything-3's
# own install needs, and nothing lives at /workspace, so runner code is always whatever is on the
# host — edits never need a rebuild. Weights are NOT downloaded at build time (see run_sequence.py's
# ensure_weights_downloaded, which fetches them lazily on first run instead, same as every other
# baseline here).

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) —
# pull a real, current 3.11 from deadsnakes instead. Everything is then pip-installed into its
# own venv (not the system python) so it can't collide with Debian-managed dist-packages
# (add-apt-repository itself pulls in distutils-installed python3-* packages). curl is for
# ensure_weights_downloaded's own curl-based downloads (run_sequence.py, not at build time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv git curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

# depth-anything-3's own deps (torch/xformers first — the slow, rarely-changing layers). torch's
# own Linux+CUDA wheel pulls in `triton` as a transitive dep, which da3_streaming's default
# Model.align_lib: 'triton' config needs -- not installed separately here for that reason.
COPY depth-anything-3/requirements.txt /tmp/da3-requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "torch>=2" torchvision xformers && \
    pip install --no-cache-dir -r /tmp/da3-requirements.txt

# Copied to /opt (not /workspace) and installed off that copy, so the import works no matter
# which host paths run_container.sh mounts and never shadows the host checkout.
COPY depth-anything-3 /opt/depth-anything-3
RUN pip install --no-cache-dir -e /opt/depth-anything-3

WORKDIR /opt/depth-anything-3/da3_streaming

# da3_streaming/requirements.txt's `faiss-gpu` has no installable wheels anymore -- same fix as
# baselines/VGGT-Long/Dockerfile (Possible Problems, faiss-gpu -> faiss-gpu-cuXX), cu12 to match
# the cu124 base image here. As documented there, these unofficial (kyamagu/faiss-wheels) wheels
# have a history of shipping without real GPU kernels compiled in -- if that matters for your run,
# see baselines/VGGT-Long/Dockerfile.vggt-env for a from-source, verified-GPU faiss build; this
# file doesn't replicate that (nobody's asked for it here yet).
#
# matplotlib is a hard, top-level import (da3_streaming.py itself: `import matplotlib.pyplot as
# plt`, for its Sim3 loop-closure optimization plot) that requirements.txt is missing -- installed
# in the *same* pip invocation as requirements.txt, not a separate one after it, so a floating
# numpy pulled in by matplotlib can't silently outrun whatever numpy version depth-anything-3's
# own requirements.txt (installed above) already pinned -- same class of bug as
# baselines/VGGT-Long/Dockerfile's own pypose/matplotlib fix, see its comment for the mechanism.
RUN sed -i 's/^faiss-gpu$/faiss-gpu-cu12/' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt matplotlib

# inspection_viz's light (non-viser-heavy) deps, for the dataset loader + output writer --
# same explicit subset (not a `-r requirements.txt` pass) as baselines/depth_anything_3/Dockerfile,
# for the same reason: trimesh comes along transitively via viser, so it's not listed here either.
# numpy is pinned here because an *installed* version doesn't stop a later, separate pip command
# from upgrading it, and opencv-python-headless's newer 4.x builds want numpy>=2 by default --
# pin both explicitly to a compatible combination.
RUN pip install --no-cache-dir \
        "numpy<2" "opencv-python-headless<5" OpenEXR pillow viser pyyaml

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[da3-streaming-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python run_sequence.py <config.yaml>` yourself (baselines/da3_streaming/run_sequence.py).
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
