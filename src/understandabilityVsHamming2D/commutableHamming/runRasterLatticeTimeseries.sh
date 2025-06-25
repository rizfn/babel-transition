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

# Parameter ranges (edit as needed)
GAMMA_VALUES=(-1 0 1 2 3 4 5)
ALPHA_VALUES=(-1 0 1 2 3 4 5)

# Fixed parameters (edit as needed)
MU=0.001
KILL_RADIUS=1
STEPS=60000

EXE=./latticeTimeseries  # Path to your executable

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

# Build the job list
JOBLIST=()
for gamma in "${GAMMA_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        JOBLIST+=("$gamma $alpha $MU $KILL_RADIUS $STEPS")
    done
done

TOTAL_JOBS=${#JOBLIST[@]}
echo "Total jobs to run: $TOTAL_JOBS"

# Export variables for xargs
export EXE

# Function to run a single job (for xargs)
run_job() {
    set -e
    gamma=$1
    alpha=$2
    mu=$3
    kill_radius=$4
    steps=$5
    echo "Starting: gamma=$gamma alpha=$alpha mu=$mu kill_radius=$kill_radius steps=$steps"
    $EXE "$gamma" "$alpha" "$mu" "$kill_radius" "$steps"
    echo "Finished: gamma=$gamma alpha=$alpha"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 5 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"