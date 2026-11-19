#!/usr/bin/env bash

# Restart slurmd sequentially from head after MPI package maintenance

set -Eeuo pipefail

NODES=(
    node00 node01 node02 node03 node04
    node05 node06 node07 node08 node09
    node10 node11 node12 node13 node14
    head
)

LOG_DIR="slurmdRestartLogs-$(date +%Y%m%d-%H%M%S)"
LOCAL_HOST="$(hostname -s)"

mkdir -p "$LOG_DIR"

# Prompt once and keep the password only in this script's memory
read -r -s -p "Sudo password: " SUDO_PASSWORD
printf "\n"

trap 'unset SUDO_PASSWORD' EXIT

read -r -p "Restart slurmd on all 16 nodes? [y/N] " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    printf "Cancelled\n"
    exit 0
fi

# Restart and verify slurmd using the password sent over SSH stdin
REMOTE_COMMAND=$(cat <<'REMOTE'
set -Eeuo pipefail

IFS= read -r sudo_password

# Authenticate this one privileged service-management command
printf '%s\n' "$sudo_password" | sudo -S -p '' systemctl restart slurmd
unset sudo_password

systemctl is-active --quiet slurmd
printf 'slurmd is active on %s\n' "$(hostname -s)"
REMOTE
)

FAILED_NODES=()

# Restart one node at a time to keep the repair process observable
for node in "${NODES[@]}"; do
    printf "\n== Restarting slurmd on %s ==\n" "$node"

    log_file="$LOG_DIR/$node.log"

    if [[ "$node" == "$LOCAL_HOST" ]]; then
        if ! printf '%s\n' "$SUDO_PASSWORD" \
            | bash -c "$REMOTE_COMMAND" 2>&1 \
            | tee "$log_file"; then
            FAILED_NODES+=("$node")
        fi
    else
        if ! printf '%s\n' "$SUDO_PASSWORD" \
            | ssh -o BatchMode=yes "$node" "$REMOTE_COMMAND" 2>&1 \
            | tee "$log_file"; then
            FAILED_NODES+=("$node")
        fi
    fi
done

if (( ${#FAILED_NODES[@]} > 0 )); then
    printf "\nFailed nodes: %s\n" "${FAILED_NODES[*]}" >&2
    printf "Review logs in %s before restarting slurmctld\n" "$LOG_DIR"
    exit 1
fi

printf "\nslurmd restarted successfully on all nodes\n"
printf "Logs are in %s\n" "$LOG_DIR"
printf "Next, restart slurmctld on head and rerun the MPI smoke test\n"
