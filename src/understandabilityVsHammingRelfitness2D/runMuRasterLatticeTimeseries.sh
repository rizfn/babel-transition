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

# Parameter ranges - now alpha varies and gamma is fixed
ALPHA_VALUES=(0 0.2 0.4 0.6 0.8 1.0 1.2)
# Exactly: 1e-4, (in between), (in between), 1e-3, (in between), (in between), 1e-2, (in between)
MU_VALUES=(0.0001 0.0002154435 0.0004641589 0.001 0.002154435 0.004641589 0.01 0.02154435)

# Fixed parameters
GAMMA=1
KILL_RADIUS=1
STEPS=60000

EXE=./latticeTimeseriesStochastic  # Path to your executable

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Mu values: ${MU_VALUES[*]}"
echo "Fixed gamma: $GAMMA"

# Build the job list
JOBLIST=()
for alpha in "${ALPHA_VALUES[@]}"; do
    for mu in "${MU_VALUES[@]}"; do
        JOBLIST+=("$GAMMA $alpha $mu $KILL_RADIUS $STEPS")
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
    echo "Finished: alpha=$alpha mu=$mu"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 5 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"