#!/usr/bin/env bash

   # Reinstall the matching MPI, PMIx, and Slurm runtime stack on every node

   set -Eeuo pipefail

   NODES=(
       head
       node00 node01 node02 node03 node04
       node05 node06 node07 node08 node09
       node10 node11 node12 node13 node14
   )

   LOG_DIR="mpiRepairLogs-$(date +%Y%m%d-%H%M%S)"
   LOCAL_HOST="$(hostname -s)"

   mkdir -p "$LOG_DIR"

   # Prompt once and keep the password only in this script's memory
   read -r -s -p "Sudo password: " SUDO_PASSWORD
   printf "\n"

   trap 'unset SUDO_PASSWORD' EXIT

   read -r -p "Reinstall MPI packages on all 16 nodes? [y/N] " CONFIRM

   if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
       printf "Cancelled\n"
       exit 0
   fi

   # This command runs remotely after receiving the password over SSH stdin
   REMOTE_COMMAND=$(cat <<'REMOTE'
   set -Eeuo pipefail

   IFS= read -r sudo_password

   # Authenticate each command because remote sudo timestamps are unavailable
   run_sudo() {
       printf '%s\n' "$sudo_password" | sudo -S -p '' "$@"
   }

   run_sudo env DEBIAN_FRONTEND=noninteractive apt-get update

   run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install \
       --reinstall \
       openmpi-bin \
       openmpi-common \
       libopenmpi40 \
       libpmix2t64 \
       libprrte3 \
       slurmd \
       slurm-wlm

   unset sudo_password
REMOTE
   )

   FAILED_NODES=()

   for node in "${NODES[@]}"; do
       printf "\n== Reinstalling packages on %s ==\n" "$node"

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
       printf "Review logs in %s before restarting services\n" "$LOG_DIR"
       exit 1
   fi

   printf "\nPackage reinstall completed on all nodes\n"
   printf "Logs are in %s\n" "$LOG_DIR"
   printf "Restart slurmd on every node, then slurmctld on head\n"
