# MP-SfM baseline: environment for cvg/mpsfm (COLMAP-style incremental SfM augmented with
# monocular depth/normal priors). Built directly on top of mpsfm's own pre-supplied Docker image
# (`mpsfm/mpsfm:latest`, per the submodule's own README "Docker" section) rather than replaying
# its from-source builder stage ourselves. That image already ships a matched, working
# Ceres/pyceres/COLMAP-fork/pycolmap build (mpsfm's own maintainers build and test it), which
# sidesteps a long list of from-source build hazards an earlier version of this Dockerfile hit
# one at a time trying to replay that build itself:
#   - a pybind11 major-version mismatch between pyceres (pinned pybind11==3.0.1) and the colmap
#     fork (pinned pybind11==2.13.6, independently) -- pybind11 2.x/3.x use incompatible internal
#     ABI tags, so pycolmap's PositiveExponentialManifold binding (which derives from pyceres's
#     own ceres::Manifold registration) failed at `import pycolmap` with increasingly specific
#     errors ("referenced unknown base type", then "does not have a non-default holder type")
#     depending which side got bumped to fix the mismatch.
#   - an LTO/ODR interaction in pycolmap's own pybind11-bindings build corrupting exception
#     handling: pycolmap/pybind11_extension.h redefines type_caster<std::string>, an ODR
#     violation the compiler itself flags -- combined with pybind11_add_module's own LTO
#     auto-enable (colmap's top-level CMakeLists.txt deliberately disables LTO for GCC builds of
#     its C++ library for exactly this reason, but that exclusion doesn't reach the separately
#     built python bindings), this miscompiled exception unwinding enough that a routine
#     exception thrown during `import pycolmap` SIGABRTed the whole interpreter instead of
#     raising a catchable Python exception.
#   - mmcv's legacy setup.py `import pkg_resources` at build time, which setuptools>=82 removed.
# All fixed at various points in the from-source variant, but the *maintainer-built* image
# sidesteps needing any of it in the first place -- these are recorded here so a future
# reasonable-looking "let's just build pyceres/pycolmap from source instead" doesn't have to
# rediscover them.
#
# Build: build_container.sh (build context is the repo root, so the submodule is copied in).
# Run: run_container.sh — it bind-mounts the host's paths onto themselves and gives you a shell;
# the repo, configs and datasets are then all at their normal host paths.
#
# Unlike this repo's other baselines, `mpsfm` itself is NOT `pip install -e`'d into the image:
# only its Python requirements are (see below). `reconstruct.py` is instead run as a subprocess
# with cwd set to the *live host* submodule checkout (same `sys.path.insert`-into-cwd trick as
# VGGT-Long/da3_streaming's own subprocess baselines) -- reconstruct.py isn't packaged to be
# imported as a library anyway (its argparse runs at module scope, unconditionally, so importing
# it as a module would just run the whole CLI as a side effect), so nothing here needs a rebuild
# after editing the submodule, and its own configs/ dir is read directly off the host mount.

FROM mpsfm/mpsfm:latest

WORKDIR /opt/mpsfm-env

# mpsfm's own requirements.txt, minus the `-e ./third_party/ml-depth-pro` line -- same build-time
# omission the base image's own build documents (that third_party submodule's own package needs
# its checkout present at install time, which isn't available in a plain `docker pull` base;
# DepthPro just isn't available as a --conf option here as a result, same as upstream's own
# image). Everything else in this file is already installed in the base image (it ships its own
# baked-in copy of this exact requirements.txt at /mpsfm/requirements.txt) -- this reinstall is a
# defensive no-op, not a fresh build, kept so the image stays self-documenting and robust if a
# future mpsfm/mpsfm:latest ever drifts out of sync with its own requirements.txt.
#
# The --build-constraint flag (not plain -c/--constraint, which pip no longer applies to
# isolated build envs) needs a newer pip than the base image ships, hence the upgrade first. It's
# dead weight today either way -- mmcv's already satisfied, so pip won't rebuild it, so its
# legacy setup.py never runs pkg_resources-importing code in the first place -- but it's cheap
# insurance against the exact failure mode already hit once, in case a future requirements.txt
# bump ever forces mmcv to actually rebuild from source.
COPY mpsfm/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v 'ml-depth-pro' requirements.txt > /tmp/req.txt && \
    printf 'setuptools<82\n' > /tmp/build-constraints.txt && \
    pip install --no-cache-dir --build-constraint /tmp/build-constraints.txt -r /tmp/req.txt

# inspection_viz's light (non-viser-heavy) deps, for the dataset loader + output writer -- same
# explicit subset as baselines/depth_anything_3/Dockerfile, for the same reason (trimesh comes
# along transitively via viser). Only OpenEXR and viser are actually missing from the base image:
# numpy/opencv-python-headless/pillow/pyyaml are already present at versions that work together
# (numpy 2.2.6, satisfying cupy-cuda12x's own numpy>=2.0 requirement) -- deliberately NOT
# re-pinning any of those here. An explicit numpy<2 pin is exactly what broke cupy in an earlier
# version of this Dockerfile ("numpy.core.multiarray failed to import" at `import cupy`, since
# cupy's own compiled Cython extension needs numpy>=2's C-API and doesn't run against 1.x) --
# don't reintroduce one without checking what's already installed first.
RUN pip install --no-cache-dir OpenEXR viser

# The base image sets ENTRYPOINT ["bash"] itself. Left as-is, `docker run ... "$IMAGE" python
# run_sequence.py ...` (run_container.sh's one-off-command mode) would compose to `bash python
# run_sequence.py ...` -- bash treats a bare non-flag first argument as a script *path* to run,
# not a command to exec, so it'd fail looking for a file literally named "python". Clearing the
# inherited entrypoint restores plain `docker run image <cmd...>` semantics (matching every other
# baseline's image in this repo), with CMD supplying the default interactive shell when
# run_container.sh is called with no command.
ENTRYPOINT []

# Interactive shells source your host ~/.bashrc (your home is mounted), which may re-order PATH
# — e.g. a conda/mamba hook — so re-assert `python`/`pip` afterwards: inside the container,
# `python` must always be this image's own /usr/bin/python (there's no separate venv here,
# matching mpsfm's own runtime-stage layout).
RUN printf '%s\n' \
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"' \
        'export PATH=/usr/local/bin:/usr/bin:$PATH' \
        'PS1="[mpsfm-container] \w \$ "' \
        > /opt/container-bashrc

# No further ENTRYPOINT: this image is an environment you get a shell in (run_container.sh),
# where you then run `python run_sequence.py <config.yaml>` yourself (baselines/mpsfm/run_sequence.py).
CMD ["bash", "--rcfile", "/opt/container-bashrc"]
