set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/pi-long}"
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
    --hostname pi-long-container \
    --user "$(id -u):$(id -g)" \
    "${GROUP_ARGS[@]}" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    "${MOUNT_ARGS[@]}" \
    -e HOME="$HOME" \
    -e TERM="${TERM:-xterm}" \
    -e HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}" \
    -e PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    -w "$HERE" \
    "$IMAGE" "$@"
