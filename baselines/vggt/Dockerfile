FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04's own `python3.11` apt package is stuck at 3.11.0rc1 (predates the final
# release) and is missing stdlib attrs recent torch needs (e.g. sys.get_int_max_str_digits) —
# pull a real, current 3.11 from deadsnakes instead, matching the `conda create -n vggt
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

WORKDIR /opt/vggt

# Install torch

# Install VGGT 
COPY vggt/ . 
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
        'PS1="[vggt-container] \w \$ "' \
        > /opt/container-bashrc

# No ENTRYPOINT: this image is an environment you get a shell in (run_container.sh), where you
# then run `python main.py --image_folder <path> ...` (or main_realtime.py) yourself.
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
