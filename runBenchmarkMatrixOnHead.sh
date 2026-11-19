#!/usr/bin/env bash

# Run selected benchmark configurations sequentially from the head node

set -Eeuo pipefail

CONFIG_FILE_NAME="benchmarkMatrix.conf"
RUNNER_FILE_NAME="runNxMAllreduceOnHead.sh"

# Print command usage and configuration selection behavior
print_usage() {
    printf 'Usage: %s [label ...]\n' "$0"
    printf 'Without labels, run every configuration in %s\n' \
        "$CONFIG_FILE_NAME"
}

# Return success when the current configuration was requested
should_run_configuration() {
    local label="$1"
    shift

    if [ "$#" -eq 0 ]; then
        return 0
    fi

    local requested_label

    for requested_label in "$@"; do
        if [ "$label" = "$requested_label" ]; then
            return 0
        fi
    done

    return 1
}

# Mark one requested configuration label as found
mark_found_configuration() {
    local label="$1"
    shift
    local index

    for index in "$@"; do
        if [ "${requested_labels[index]}" = "$label" ]; then
            found_labels[index]=1
        fi
    done
}

if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    print_usage
    exit 0
fi

if [ "$(hostname -s)" != "head" ]; then
    printf 'This script must run on head\n' >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
config_file="$script_dir/$CONFIG_FILE_NAME"
runner_file="$script_dir/$RUNNER_FILE_NAME"
requested_labels=("$@")
found_labels=()

if [ ! -f "$config_file" ]; then
    printf 'Missing configuration file: %s\n' "$config_file" >&2
    exit 1
fi

if [ ! -x "$runner_file" ]; then
    printf 'Missing executable runner: %s\n' "$runner_file" >&2
    exit 1
fi

for index in "${!requested_labels[@]}"; do
    found_labels[index]=0
done

# Keep generated matrix logs beside the benchmark CSV result files
mkdir -p "$data_dir"
log_file="$data_dir/benchmarkMatrix-$(date +%Y%m%d-%H%M%S).log"
printf 'Benchmark matrix started: %s\n' "$(date --iso-8601=seconds)" \
    | tee "$log_file"
printf 'Configuration file: %s\n' "$config_file" | tee -a "$log_file"

while read -r label node_count ranks_per_node node_list extra; do
    if [ -z "${label:-}" ] || [[ "$label" = \#* ]]; then
        continue
    fi

    if [ -z "${node_count:-}" ] || [ -z "${ranks_per_node:-}" ] \
        || [ -n "${extra:-}" ]; then
        printf 'Invalid configuration line for %s\n' "$label" >&2
        exit 1
    fi

    if ! should_run_configuration "$label" "${requested_labels[@]}"; then
        continue
    fi

    mark_found_configuration "$label" "${!requested_labels[@]}"

    printf '\n== Starting %s at %s ==\n' \
        "$label" "$(date --iso-8601=seconds)" | tee -a "$log_file"
    if [ -n "${node_list:-}" ] && [ "$node_list" != "-" ]; then
        selection_description="$node_list"
    elif [ "$node_count" -eq 16 ]; then
        selection_description="all cluster nodes"
    else
        selection_description="random compute nodes"
    fi

    printf 'nodes=%s ranksPerNode=%s nodeList=%s\n' \
        "$node_count" "$ranks_per_node" "$selection_description" \
        | tee -a "$log_file"

    if [ -n "${node_list:-}" ] && [ "$node_list" != "-" ]; then
        NXM_NODELIST="$node_list" "$runner_file" \
            "$node_count" "$ranks_per_node" 2>&1 | tee -a "$log_file"
    else
        "$runner_file" "$node_count" "$ranks_per_node" 2>&1 \
            | tee -a "$log_file"
    fi

    printf '== Completed %s at %s ==\n' \
        "$label" "$(date --iso-8601=seconds)" | tee -a "$log_file"
done < "$config_file"

for index in "${!requested_labels[@]}"; do
    if [ "${found_labels[index]}" -ne 1 ]; then
        printf 'Unknown configuration label: %s\n' \
            "${requested_labels[index]}" >&2
        exit 1
    fi
done

printf '\nBenchmark matrix completed: %s\n' \
    "$(date --iso-8601=seconds)" | tee -a "$log_file"
printf 'Log: %s\n' "$log_file"
