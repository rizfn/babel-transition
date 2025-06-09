#!/bin/bash
# filepath: run_latticeNbrVsGlobal_parallel.sh

# Number of CPUs to leave free
CPUS_FREE=2

# Get total logical CPUs
TOTAL_CPUS=$(nproc)
MAX_PARALLEL=$((TOTAL_CPUS - CPUS_FREE))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    MAX_PARALLEL=1
fi

echo "Total CPUs: $TOTAL_CPUS, Running with $MAX_PARALLEL parallel jobs (leaving $CPUS_FREE free)"

# Parameter ranges (edit as needed)
GAMMA_VALUES=(0 1 2 3 4 5 6)
ALPHA_VALUES=(0 1 2 3 4 5 6)
GIR_VALUES=(0.5 1.0 2.0)  # Example global_interaction_ratio values

# Fixed parameters (edit as needed)
MU=0.01
KILL_RADIUS=3
STEPS=1000

EXE=./latticeNbrVsGlobal  # Path to your executable

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

# Build the job list
JOBLIST=()
for gamma in "${GAMMA_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        for gir in "${GIR_VALUES[@]}"; do
            JOBLIST+=("$gamma $alpha $gir $MU $KILL_RADIUS $STEPS")
        done
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
    gir=$3
    mu=$4
    kill_radius=$5
    steps=$6
    echo "Starting: gamma=$gamma alpha=$alpha gir=$gir mu=$mu kill_radius=$kill_radius steps=$steps"
    $EXE "$gamma" "$alpha" "$gir" "$mu" "$kill_radius" "$steps"
    echo "Finished: gamma=$gamma alpha=$alpha gir=$gir"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 6 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"