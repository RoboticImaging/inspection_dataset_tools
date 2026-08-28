#!/usr/bin/env bash
# Drop into an interactive shell inside the glomap baseline image. That's all it does — it runs
# no baseline itself. Once inside:
#
#   python run_sequence.py <config.yaml>          # cwd starts at this baseline's folder
#
# `colmap` itself (including `colmap global_mapper`, GLOMAP's global SfM pipeline — see the
# Dockerfile's own header) is also usable directly as a CLI from that same shell, e.g.:
#   colmap global_mapper --database_path db.db --image_path ./images --output_path ./sparse
# or any other colmap command — see `colmap --help`, or colmap/doc/ for the full reference.
#
# Usage:
#   baselines/glomap/run_container.sh          # shell
#   baselines/glomap/run_container.sh <cmd...> # run one command, then exit
#
# Env overrides: IMAGE, GPUS (default all, e.g. GPUS='"device=1"'), MOUNTS (space-separated
# extra host paths), SHM_SIZE.
#
# The container sees the host's filesystem at the *same paths* it has outside (/home, /mnt,
# ... are bind-mounted onto themselves), so every absolute path — the repo, config yamls,
# dataset roots — is valid inside without translation, and no rebuild is needed after editing
# code. It runs as your uid/gid (with your supplementary groups, and /etc/passwd + /etc/group
# mounted read-only so the names resolve), so whatever COLMAP/run_sequence.py writes is owned by
# you and is deletable from outside the container.
#
# No HF_HOME (unlike this repo's feed-forward baselines): GLOMAP is classical SfM/MVS, not a
# downloaded checkpoint -- nothing here ever touches HuggingFace.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/glomap}"
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
    --hostname glomap-container \
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
