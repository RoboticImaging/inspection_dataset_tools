# Depth Anything 3 baseline: runs the `depth-anything-3` submodule against a sequence from
# `inspection_viz.dataset` and writes depth + camera predictions back into the asset folder.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a
# shell; the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment (python + deps + the depth-anything-3 submodule
# installed at /opt). No repo code is baked in and nothing lives at /workspace, so runner code
# is always whatever is on the host — edits never need a rebuild.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
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

# depth-anything-3's own deps (torch/xformers first — the slow, rarely-changing layers)
COPY depth-anything-3/requirements.txt /tmp/da3-requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "torch>=2" torchvision xformers && \
    pip install --no-cache-dir -r /tmp/da3-requirements.txt

# Copied to /opt (not /workspace) and installed off that copy, so the import works no matter
# which host paths run_container.sh mounts and never shadows the host checkout.
COPY depth-anything-3 /opt/depth-anything-3
RUN pip install --no-cache-dir -e /opt/depth-anything-3

# inspection_viz's light (non-viser-heavy) deps, for the dataset loader + output writer.
# depth-anything-3/requirements.txt pinned numpy<2 above; pip's resolver only respects that as a
# hard constraint if we restate it here too (an *installed* version doesn't stop it upgrading),
# and even opencv-python-headless's newer 4.x builds now want numpy>=2 by default, so pin both
# explicitly to a combination that's actually compatible.
# inspection_viz itself is not installed — run_sequence.py imports it from the host checkout.
RUN pip install --no-cache-dir \
        "numpy<2" "opencv-python-headless<5" OpenEXR pillow viser pyyaml

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[da3-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python run_sequence.py <config.yaml>` yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
