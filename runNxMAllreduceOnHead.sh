#!/usr/bin/env bash

# Submit an N-by-M benchmark and collect a validated CSV on head

set -Eeuo pipefail

POLL_SECONDS=5
MIN_ELEMENTS=1000
MAX_ELEMENTS=150000
ELEMENT_STEP=1000
CSV_HEADER='elements,payload_bytes,trial,mpi_us_op,custom_us_op,'
CSV_HEADER+='mpi_correct,custom_correct,mpi_first,world_ranks,node_count'

# Print runner usage and the required execution location
print_usage() {
    printf 'Usage: %s N M [minimumElements maximumElements]\n' "$0"
    printf 'Set NXM_NODELIST to pin exactly N nodes when needed\n'
    printf 'Run this script from ~/adaptive-allreduce-test on head\n'
}

# Read a Slurm job state from its retained controller record
get_job_field() {
    local job_id="$1"
    local field_name="$2"

    scontrol show job "$job_id" \
        | awk -F= -v field_name="$field_name" '
            $0 ~ field_name "=" {
                split($2, fields, " ")
                print fields[1]
                exit
            }
        '
}

# Copy a batch-host file to stdout without requiring shared home storage
read_batch_file() {
    local batch_host="$1"
    local file_path="$2"
    local local_host

    local_host="$(hostname -s)"

    if [ "$batch_host" = "$local_host" ]; then
        cat "$file_path"
    else
        ssh -o BatchMode=yes "$batch_host" "cat '$file_path'"
    fi
}

# Remove batch-host output files after a successful CSV collection
remove_batch_files() {
    local batch_host="$1"
    local output_path="$2"
    local error_path="$3"
    local local_host

    local_host="$(hostname -s)"

    if [ "$batch_host" = "$local_host" ]; then
        rm -f "$output_path" "$error_path"
    else
        ssh -o BatchMode=yes "$batch_host" \
            "rm -f '$output_path' '$error_path'"
    fi
}

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    print_usage
    exit 1
fi

if [ "$(hostname -s)" != "head" ]; then
    printf 'This script must run on head\n' >&2
    exit 1
fi

node_count="$1"
ranks_per_node="$2"
minimum_elements="${3:-$MIN_ELEMENTS}"
maximum_elements="${4:-$MAX_ELEMENTS}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submit_script="$script_dir/submitNxMAllreduce.sh"

if [ ! -x "$submit_script" ]; then
    printf 'Missing executable submit script: %s\n' "$submit_script" >&2
    exit 1
fi

submit_output="$("$submit_script" \
    "$node_count" \
    "$ranks_per_node" \
    "$minimum_elements" \
    "$maximum_elements")"
printf '%s\n' "$submit_output"

job_id="$(printf '%s\n' "$submit_output" \
    | awk '/^Submitted job [0-9]+ / { print $3; exit }')"

if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Could not determine the submitted Slurm job ID\n' >&2
    exit 1
fi

printf 'Waiting for job %s to complete\n' "$job_id"

while squeue -h -j "$job_id" | grep -q .; do
    sleep "$POLL_SECONDS"
done

job_state="$(get_job_field "$job_id" "JobState")"

if [ "$job_state" != "COMPLETED" ]; then
    printf 'Job %s ended with state %s\n' "$job_id" "$job_state" >&2
    exit 1
fi

batch_host="$(get_job_field "$job_id" "BatchHost")"
output_path="/tmp/nxm-allreduce-$job_id.out"
error_path="/tmp/nxm-allreduce-$job_id.err"
output_prefix="nodes${node_count}ranks${ranks_per_node}Allreduce-$job_id"
csv_path="$script_dir/$output_prefix.csv"
expected_rows=$((
    ((maximum_elements - minimum_elements) / ELEMENT_STEP + 1) * 100
))
total_ranks=$((node_count * ranks_per_node))

printf '%s\n' "$CSV_HEADER" > "$csv_path"

read_batch_file "$batch_host" "$output_path" \
    | awk '
        {
            line = $0
            sub(/^[[:space:]]*[0-9]+: /, "", line)

            if (line !~ /^elements,payload/) {
                print line
            }
        }
    ' >> "$csv_path"

awk -F, \
    -v expected_rows="$expected_rows" \
    -v expected_ranks="$total_ranks" \
    -v expected_nodes="$node_count" '
    NR == 1 {
        if ($1 != "elements" || $10 != "node_count") {
            invalid = 1
        }

        next
    }

    {
        rows += 1

        if (NF != 10 || $6 != "1" || $7 != "1") {
            invalid = 1
        }

        if ($9 != expected_ranks || $10 != expected_nodes) {
            invalid = 1
        }
    }

    END {
        if (invalid || rows != expected_rows) {
            exit 1
        }
    }
' "$csv_path"

remove_batch_files "$batch_host" "$output_path" "$error_path"

printf 'Collected and validated %d rows\n' "$expected_rows"
printf 'CSV: %s\n' "$csv_path"
