#!/bin/bash

# Number of CPUs to leave free
CPUS_FREE=2

# Get total logical CPUs
TOTAL_CPUS=$(nproc)
MAX_PARALLEL=$((TOTAL_CPUS - CPUS_FREE))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    MAX_PARALLEL=1
fi

echo "Total CPUs: $TOTAL_CPUS, Running with max $MAX_PARALLEL parallel jobs (leaving $CPUS_FREE free)"

# Parameter ranges - both alpha and gamma vary now
GAMMA_VALUES=(-1 0 1 2 3 4 5)
ALPHA_VALUES=(-1 0 1 2 3 4 5)

# Fixed parameters
L=1024
B=16
MU=0.001  # Fixed mu value
STEPS=60000
STEPS_TO_RECORD=40000
RECORDING_SKIP=50

EXE=./latticeTimeseries

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

echo "Fixed parameters: L=$L, B=$B, mu=$MU, steps=$STEPS, steps_to_record=$STEPS_TO_RECORD, recording_skip=$RECORDING_SKIP"
echo "Gamma values: ${GAMMA_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"

# Build the job list
JOBLIST=()
for gamma in "${GAMMA_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        JOBLIST+=("$L $B $STEPS $gamma $alpha $MU $STEPS_TO_RECORD $RECORDING_SKIP")
    done
done

TOTAL_JOBS=${#JOBLIST[@]}
echo "Total jobs to run: $TOTAL_JOBS"

# Export variables for xargs
export EXE

# Function to run a single job (for xargs)
run_job() {
    set -e
    L=$1
    B=$2
    steps=$3
    gamma=$4
    alpha=$5
    mu=$6
    steps_to_record=$7
    recording_skip=$8
    echo "Starting: L=$L B=$B steps=$steps gamma=$gamma alpha=$alpha mu=$mu steps_to_record=$steps_to_record recording_skip=$recording_skip"
    $EXE "$L" "$B" "$steps" "$gamma" "$alpha" "$mu" "$steps_to_record" "$recording_skip"
    echo "Finished: gamma=$gamma alpha=$alpha"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 8 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"