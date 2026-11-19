#!/usr/bin/env bash

# Submit an N-node by M-rank-per-node benchmark from the head node

set -Eeuo pipefail

MAX_NODES=16
COMPUTE_NODE_COUNT=15
MAX_PHYSICAL_RANKS=4
MIN_ELEMENTS=1000
MAX_ELEMENTS=150000
ELEMENT_STEP=1000
DEFAULT_RANGE_WIDTH=10000
DEFAULT_TIME_LIMIT="02:00:00"
POLL_SECONDS=2

# Print command usage and supported benchmark limits
print_usage() {
    printf 'Usage: %s N M [minimumElements maximumElements]\n' "$0"
    printf '  N: requested node count from 1 through %d\n' "$MAX_NODES"
    printf '  M: ranks per node from 1 through %d\n' \
        "$MAX_PHYSICAL_RANKS"
    printf '  Element bounds default to %d through %d\n' \
        "$MIN_ELEMENTS" "$MAX_ELEMENTS"
    printf '  Set NXM_NODELIST to pin exactly N nodes\n'
    printf '  Without it, select random compute nodes\n'
}

# Require a positive base-10 integer argument
require_positive_integer() {
    local value="$1"
    local label="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        printf '%s must be a positive integer\n' "$label" >&2
        exit 1
    fi
}

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    print_usage
    exit 1
fi

node_count="$1"
ranks_per_node="$2"
minimum_elements="${3:-$MIN_ELEMENTS}"
maximum_elements="${4:-$MAX_ELEMENTS}"

require_positive_integer "$node_count" "N"
require_positive_integer "$ranks_per_node" "M"
require_positive_integer "$minimum_elements" "minimumElements"
require_positive_integer "$maximum_elements" "maximumElements"

if [ "$node_count" -gt "$MAX_NODES" ]; then
    printf 'N cannot exceed %d available nodes\n' "$MAX_NODES" >&2
    exit 1
fi

if [ "$ranks_per_node" -gt "$MAX_PHYSICAL_RANKS" ]; then
    printf 'M cannot exceed %d physical ranks per node\n' \
        "$MAX_PHYSICAL_RANKS" >&2
    exit 1
fi

if [ "$minimum_elements" -lt "$MIN_ELEMENTS" ] \
    || [ "$maximum_elements" -gt "$MAX_ELEMENTS" ] \
    || [ "$minimum_elements" -gt "$maximum_elements" ]; then
    printf 'Element bounds must be between %d and %d\n' \
        "$MIN_ELEMENTS" "$MAX_ELEMENTS" >&2
    exit 1
fi

if [ $((minimum_elements % ELEMENT_STEP)) -ne 0 ] \
    || [ $((maximum_elements % ELEMENT_STEP)) -ne 0 ]; then
    printf 'Element bounds must be divisible by %d\n' \
        "$ELEMENT_STEP" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="$script_dir/rangeAllreduce.cpp"
binary_file="$script_dir/rangeAllreduce"
batch_file="$script_dir/runNxMAllreduce.sbatch"
total_ranks=$((node_count * ranks_per_node))
time_limit="${NXM_TIME_LIMIT:-$DEFAULT_TIME_LIMIT}"
node_list="${NXM_NODELIST:-}"
node_selection="provided"
node_options=()

if [ ! -f "$source_file" ] || [ ! -f "$batch_file" ]; then
    printf 'Run this script from the benchmark directory on head\n' >&2
    exit 1
fi

# Select compute nodes when an explicit Slurm host list was not provided
if [ -z "$node_list" ]; then
    if [ "$node_count" -eq "$MAX_NODES" ]; then
        node_list="head,node[00-14]"
        node_selection="all nodes"
    else
        random_candidates=()

        for node_index in $(seq 0 $((COMPUTE_NODE_COUNT - 1))); do
            printf -v candidate_node 'node%02d' "$node_index"
            random_candidates+=("$candidate_node")
        done

        node_list="$(printf '%s\n' "${random_candidates[@]}" \
            | shuf -n "$node_count" | paste -sd, -)"
        node_selection="random compute nodes"
    fi
fi

# Expand and validate the explicit or randomly selected Slurm host list
if ! expanded_nodes="$(scontrol show hostnames "$node_list")"; then
    printf 'Invalid node list: %s\n' "$node_list" >&2
    exit 1
fi

selected_node_count="$(printf '%s\n' "$expanded_nodes" \
    | awk 'NF { count += 1 } END { print count + 0 }')"

if [ "$selected_node_count" -ne "$node_count" ]; then
    printf 'Node list contains %d nodes but N is %d\n' \
        "$selected_node_count" "$node_count" >&2
    exit 1
fi

node_options+=(--nodelist="$node_list")

# Compile on head because compute nodes do not share the home directory
mpicxx -std=c++17 -O2 -Wall -Wextra -pedantic \
    "$source_file" -o "$binary_file"

export_values="ALL,NXM_MIN=$minimum_elements,"
export_values+="NXM_MAX=$maximum_elements,"
export_values+="NXM_WIDTH=$DEFAULT_RANGE_WIDTH"

job_id="$(sbatch --parsable \
    --nodes="$node_count" \
    --ntasks="$total_ranks" \
    --ntasks-per-node="$ranks_per_node" \
    --cpus-per-task=1 \
    --hint=nomultithread \
    --time="$time_limit" \
    "${node_options[@]}" \
    --export="$export_values" \
    "$batch_file")"

printf 'Submitted job %s for %d nodes and %d ranks per node\n' \
    "$job_id" "$node_count" "$ranks_per_node"

printf 'Node selection: %s\n' "$node_selection"
printf 'Requested nodes: %s\n' "$node_list"

# Wait for an allocation because sbcast requires a running Slurm job
while true; do
    state="$(squeue -h -j "$job_id" -o '%T' | tr -d '[:space:]')"

    if [ "$state" = "RUNNING" ]; then
        break
    fi

    if [ -z "$state" ]; then
        printf 'Job %s ended before binary staging\n' "$job_id" >&2
        exit 1
    fi

    sleep "$POLL_SECONDS"
done

# Copy the head-node executable to local storage on every allocated node
sbcast --preserve --jobid="$job_id" \
    "$binary_file" "/tmp/rangeAllreduce-$job_id"

batch_host="$(scontrol show job "$job_id" \
    | awk -F= '/BatchHost=/{split($2, fields, " "); print fields[1]}')"

printf 'Staged executable for job %s\n' "$job_id"
printf 'Batch host: %s\n' "$batch_host"
printf 'Monitor: squeue -j %s\n' "$job_id"
printf 'Output: /tmp/nxm-allreduce-%s.out on %s\n' \
    "$job_id" "$batch_host"
