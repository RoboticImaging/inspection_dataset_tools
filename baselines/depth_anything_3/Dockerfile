# Depth Anything 3 baseline: runs the `depth-anything-3` submodule against a sequence from
# `inspection_viz.dataset` and writes depth + camera predictions back into the asset folder.
#
# Build (from repo root, so both the submodule and inspection_viz are in the build context):
#   docker build -f baselines/depth_anything_3/Dockerfile -t inspection-baselines/da3 .
# Run: see run_container.sh — it mounts the repo (for live edits) and the dataset root.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) —
# pull a real, current 3.11 from deadsnakes instead. Everything is then pip-installed into its
# own venv (not the system python) so it can't collide with Debian-managed dist-packages
# (add-apt-repository itself pulls in distutils-installed python3-* packages).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

WORKDIR /workspace

# depth-anything-3's own deps (torch/xformers first — the slow, rarely-changing layers)
COPY depth-anything-3/requirements.txt /tmp/da3-requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "torch>=2" torchvision xformers && \
    pip install --no-cache-dir -r /tmp/da3-requirements.txt

COPY depth-anything-3 /workspace/depth-anything-3
RUN pip install --no-cache-dir -e /workspace/depth-anything-3

# inspection_viz's light (non-viser-heavy) deps, for the dataset loader + output writer.
# depth-anything-3/requirements.txt pinned numpy<2 above; pip's resolver only respects that as a
# hard constraint if we restate it here too (an *installed* version doesn't stop it upgrading),
# and even opencv-python-headless's newer 4.x builds now want numpy>=2 by default, so pin both
# explicitly to a combination that's actually compatible.
# The rest of the repo (including inspection_viz itself) is bind-mounted at run time by
# run_container.sh, so code edits don't require a rebuild.
RUN pip install --no-cache-dir \
        "numpy<2" "opencv-python-headless<5" OpenEXR pillow viser pyyaml

ENTRYPOINT ["python3", "baselines/depth_anything_3/run_sequence.py"]
