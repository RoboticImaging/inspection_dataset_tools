#!/usr/bin/env bash
# Drop into an interactive shell inside the pixel-perfect-sfm baseline image. That's all it does
# — it runs no baseline itself. Once inside:
#
#   python run_sequence.py <config.yaml>          # cwd starts at this baseline's folder
#
# `colmap` itself is also usable directly as a CLI from that same shell (e.g. `colmap
# patch_match_stereo ...`), and pixsfm's own Python API (`import pixsfm`) works from anywhere --
# see `colmap --help` / colmap/doc/, and pixel-perfect-sfm/README.md / demo.ipynb, respectively.
#
# Usage:
#   baselines/pixel-perfect-sfm/run_container.sh          # shell
#   baselines/pixel-perfect-sfm/run_container.sh <cmd...> # run one command, then exit
#
# Env overrides: IMAGE, GPUS (default all, e.g. GPUS='"device=1"'), MOUNTS (space-separated
# extra host paths), SHM_SIZE.
#
# The container sees the host's filesystem at the *same paths* it has outside (/home, /mnt,
# ... are bind-mounted onto themselves), so every absolute path — the repo, config yamls,
# dataset roots — is valid inside without translation, and no rebuild is needed after editing
# baselines/pixel-perfect-sfm/run_sequence.py. UNLIKE baselines/mpsfm, though, the
# pixel-perfect-sfm submodule itself is NOT bind-mounted over its copy in the image: the
# Dockerfile COPYs it in and builds pixsfm's own C++ extension (`pip install -e .`) in place
# there, so the compiled `_pixsfm*.so` lives only inside the image's own /pixel-perfect-sfm, not
# on the host's checkout -- shadowing it with a bind mount would break `import pixsfm`. Edit
# pixel-perfect-sfm/ itself and you need build_container.sh again, same as pixsfm's own C++
# sources always have.
#
# No HF_HOME (unlike this repo's other from-source SfM baselines, e.g. mpsfm): pixsfm's own dense
# feature extractor (S2DNet) fetches its weights from Dropbox, not a HuggingFace cache -- see
# pixsfm/features/models/s2dnet.py. Baked into the image at build time (the Dockerfile's own
# comment on that step explains why: this container's non-root user, see above, can't write into
# that root-owned directory at runtime), so nothing needs fetching here regardless of container
# user or how often you run this baseline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/pixel-perfect-sfm}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-8g}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "image $IMAGE isn't built yet — run ${HERE}/build_container.sh first" >&2
    exit 1
}

# Host trees to replicate. /home covers the repo and most datasets; /mnt and /media cover the
# extra data disks. Anything else (another mount, a nfs export) goes in $MOUNTS.
MOUNT_ARGS=()
for path in /home /mnt /media ${MOUNTS:-}; do
    [ -d "$path" ] && MOUNT_ARGS+=(-v "${path}:${path}")
done

# Your supplementary groups too, so group-readable dataset dirs stay readable in here.
GROUP_ARGS=()
for gid in $(id -G); do GROUP_ARGS+=(--group-add "$gid"); done

# -t only when there's a real terminal, so `run_container.sh <cmd>` also works from a script
# or a pipe (docker refuses to attach a TTY to a non-terminal stdin).
TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS+=(-t)

exec docker run --rm "${TTY_ARGS[@]}" \
    --gpus "$GPUS" \
    --shm-size "$SHM_SIZE" \
    --hostname pixel-perfect-sfm-container \
    --user "$(id -u):$(id -g)" \
    "${GROUP_ARGS[@]}" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    "${MOUNT_ARGS[@]}" \
    -e HOME="$HOME" \
    -e TERM="${TERM:-xterm}" \
    -e PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    -w "$HERE" \
    "$IMAGE" "$@"
