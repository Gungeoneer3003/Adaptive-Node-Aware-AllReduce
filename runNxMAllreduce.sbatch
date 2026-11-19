#!/bin/bash
#SBATCH --job-name=nxm-allreduce
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/nxm-allreduce-%j.out
#SBATCH --error=/tmp/nxm-allreduce-%j.err

# Run one range benchmark after submitNxMAllreduce.sh stages the binary

set -Eeuo pipefail

export OMPI_MCA_btl=self,sm,tcp
export OMPI_MCA_btl_tcp_if_include=192.168.88.0/24

binary="/tmp/rangeAllreduce-${SLURM_JOB_ID}"
range_min="${NXM_MIN:?NXM_MIN must be set by the submit script}"
range_max="${NXM_MAX:?NXM_MAX must be set by the submit script}"
range_width="${NXM_WIDTH:-10000}"

# Remove staged copies from every allocated node when the job exits
cleanup_binary() {
    srun bash -c 'rm -f "$1"' _ "$binary" || true
}

trap cleanup_binary EXIT

# Gives the submit wrapper time to call sbcast from head
sleep 30

while [ "$range_min" -le "$range_max" ]; do
    range_end=$((range_min + range_width - 1000))

    if [ "$range_end" -gt "$range_max" ]; then
        range_end="$range_max"
    fi

    srun --label --cpu-bind=cores env \
        OMPI_MCA_btl=self,sm,tcp \
        OMPI_MCA_btl_tcp_if_include=192.168.88.0/24 \
        "$binary" "$range_min" "$range_end"

    range_min=$((range_end + 1000))
done
