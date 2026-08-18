# Pi3X baseline: same submodule as ../Pi3 (yyfz/Pi3 vendors both `pi3.models.pi3.Pi3` and
# `pi3.models.pi3x.Pi3X` + `pi3.pipe.pi3x_vo.Pi3XVO` in one package) -- no separate git submodule,
# just a different baseline entry point (run_sequence.py) built on the same source. This file is
# ../Pi3/Dockerfile line-for-line otherwise; see that one's comments for the apt/venv rationale.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) —
# pull a real, current 3.11 from deadsnakes instead, matching the `conda create -n Pi3
# python=3.11` the upstream README calls for. Everything is then pip-installed into its own venv
# (not the system python) so it can't collide with Debian-managed dist-packages (add-apt-
# repository itself pulls in distutils-installed python3-* packages).
#
# build-essential is here (unlike the other baselines' Dockerfiles) because salad/Pi3 fork/PE/
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

WORKDIR /opt/Pi3

# Install Pi3 (package also holds Pi3X -- see header)
COPY Pi3/ .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Install inspection_dataset_tools requirements
COPY requirements.txt ./requirements_idt.txt
RUN pip install --no-cache-dir -r ./requirements_idt.txt

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert the image's venv afterwards: inside the container,
# `python` must always be /opt/venv/bin/python.
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/opt/venv/bin:$PATH' \
        'PS1="[Pi3X-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python run_sequence.py <config.yaml>` yourself (baselines/Pi3X/run_sequence.py).
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
