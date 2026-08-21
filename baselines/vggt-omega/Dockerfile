# vggt-omega baseline: environment for facebookresearch/vggt-omega (plain feed-forward camera +
# depth reconstruction, VGGT's successor).
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# The image holds only the environment (python + vggt-omega's pip deps, editable-installed). No
# repo code is baked in beyond what that install needs, and nothing lives at /workspace, so runner
# code is always whatever is on the host — edits never need a rebuild. Weights are NOT downloaded
# at build time (see the note near the bottom) -- and unlike every other baseline here, they can't
# be fetched anonymously even at runtime: the checkpoints are a gated HuggingFace repo
# (facebook/VGGT-Omega) that needs a manually-approved access request.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) --
# pull a real, current 3.11 from deadsnakes instead, same as baselines/vggt/Dockerfile (this is
# VGGT's own successor, with no compiled extensions of its own -- so it shares that Dockerfile's
# shape almost exactly, just a different repo and an explicit torch pin below where vggt/Dockerfile
# instead relies on its own requirements.txt's exact `torch==2.3.1`). Everything
# is then pip-installed into its own venv (not the system python) so it can't collide with
# Debian-managed dist-packages (add-apt-repository itself pulls in distutils-installed python3-*
# packages).
#
# build-essential is here defensively (vggt-omega's own package has no C/Cython extensions to
# build, but pip may still need it to build sdists for a transitive dependency without a wheel for
# this exact python/platform combo). libsm6/libxext6/libxrender1 are for `opencv-python`
# (requirements.txt pulls the GUI-enabled build, not -headless), on top of the libgl1/libglib2.0-0
# cv2 needs just to import.
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

WORKDIR /opt/vggt-omega

# Pinned torch first, same "Step 1" shape as baselines/VGGT-Long and baselines/Pi-Long's own
# Dockerfiles -- vggt-omega/requirements.txt only floors torch at `torch>=2.3` (see the note in
# the apt RUN above), so an unpinned `pip install -r requirements.txt` would let pip resolve
# whatever's newest on PyPI. Today that's a build whose default Linux wheel bundles CUDA 13.0
# (via a `cuda-toolkit` dependency), which needs a newer driver than most hosts have and fails at
# import with "The NVIDIA driver on your system is too old". Installing an exact, cu128-indexed
# torch/torchvision here first satisfies that floor, so requirements.txt's own install below
# leaves them alone. cu128 matches what `nvidia-smi` reports as the driver's max supported CUDA
# version here; bump only after confirming the host driver supports it.
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Install vggt-omega
COPY vggt-omega/ .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# huggingface_hub isn't a declared dependency of vggt-omega itself (its README has you download
# checkpoints by hand from the HF web UI), but its checkpoints are gated -- run_sequence.py's own
# ensure_checkpoint_downloaded fetches them via `hf download` instead, which needs this installed.
# `hf` respects $HF_TOKEN / a cached `hf auth login`, either of which still first needs your HF
# access request to facebook/VGGT-Omega approved.
RUN pip install --no-cache-dir huggingface_hub

# Install inspection_dataset_tools requirements
COPY requirements.txt ./requirements_idt.txt
# vggt-omega's own requirements.txt pins `numpy<2`; this shared, unpinned requirements.txt has no
# memory of that pin (by design -- most baselines don't care), and opencv-python-headless/viser's
# own floor requirements could pull numpy up past it if not re-asserted here. Same class of fix as
# baselines/VGGT-Long and baselines/Pi-Long's own numpy re-pins, just precautionary here rather
# than tied to a specific known crash: nothing in vggt-omega actually carries a numpy-1.x-ABI
# compiled extension the way those two's numba pin does, but there's no reason to drift outside
# what upstream declares as supported either.
RUN pip install --no-cache-dir -r ./requirements_idt.txt "numpy<2"

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[vggt-omega-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python run_sequence.py --data ... --model ...` yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
