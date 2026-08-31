#!/usr/bin/env bash
# Drop into an interactive shell inside the DetectorFreeSfM baseline image. That's all it does —
# it runs no baseline itself. Once inside (default cwd is this directory, same convention as
# every other baseline here):
#
#   python run_sequence.py --data ../configs/data/example.yaml --model ../configs/detector-free-sfm.yaml
#
# `run_sequence.py` drives DetectorFreeSfM/eval_dataset.py as a subprocess (see that file's own
# module docstring for why, and for the manual weight/ prerequisite it needs first). To instead
# use DetectorFreeSfM's own CLI directly, bypassing this repo's data/output conventions entirely
# (its README's own documented usage, paths as they are on the host -- see below):
#   cd ../../DetectorFreeSfM && python eval_dataset.py +demo=dfsfm.yaml
#
# `colmap` on PATH inside the container is the MODIFIED fork this baseline's Dockerfile builds
# from github.com/hxy-123/colmap, not vanilla COLMAP — see the Dockerfile's own header.
#
# Usage:
#   baselines/DetectorFreeSfM/run_container.sh          # shell
#   baselines/DetectorFreeSfM/run_container.sh <cmd...> # run one command, then exit
#
# Env overrides: IMAGE, GPUS (default all, e.g. GPUS='"device=1"'), MOUNTS (space-separated extra
# host paths), SHM_SIZE.
#
# The container sees the host's filesystem at the *same paths* it has outside (/home, /mnt, ...
# are bind-mounted onto themselves), so every absolute path — the repo, hydra configs, dataset
# roots, the manually-downloaded weight/ dir — is valid inside without translation, and no rebuild
# is needed after editing DetectorFreeSfM/src, its hydra_configs/, or this directory's own
# run_sequence.py. UNLIKE baselines/pixel-perfect-sfm, this baseline's own submodule was never
# COPY'd into the image in the first place (see the Dockerfile's own header) — there's no
# shadowing risk to worry about here, and no build_container.sh re-run needed for anything except
# third_party/RoIAlign.pytorch or the Dockerfile itself.
#
# PYTHONPATH is set to the repo root, same as every other baseline here — `baselines.common` and
# `inspection_viz` (both used by run_sequence.py) are imported from the host checkout, not
# installed into the image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/detector-free-sfm}"
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

# -t only when there's a real terminal, so `run_container.sh <cmd>` also works from a script or a
# pipe (docker refuses to attach a TTY to a non-terminal stdin).
TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS+=(-t)

exec docker run --rm "${TTY_ARGS[@]}" \
    --gpus "$GPUS" \
    --shm-size "$SHM_SIZE" \
    --hostname detector-free-sfm-container \
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
