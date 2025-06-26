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

# Parameter ranges to vary
GAMMA_VALUES=(3)
ALPHA_VALUES=(-1 -0.5 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5)

# Fixed parameters - define all constants here
L=256                    # lattice size
B=16                     # bitstring length
N_STEPS=100000           # total simulation steps
MU=0.001                # mutation rate
KILL_RADIUS=1           # kill radius
STEPS_TO_RECORD=50000   # number of steps to record at the end
RECORDING_SKIP=100       # record every N steps

EXE=./clusterSizesTimeseries  # Path to your executable

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

# Build the job list
JOBLIST=()
for gamma in "${GAMMA_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        JOBLIST+=("$L $B $N_STEPS $gamma $alpha $MU $KILL_RADIUS $STEPS_TO_RECORD $RECORDING_SKIP")
    done
done

TOTAL_JOBS=${#JOBLIST[@]}
echo "Total jobs to run: $TOTAL_JOBS"
echo "Parameters: L=$L, B=$B, N_STEPS=$N_STEPS, MU=$MU, KILL_RADIUS=$KILL_RADIUS"
echo "Recording: STEPS_TO_RECORD=$STEPS_TO_RECORD, RECORDING_SKIP=$RECORDING_SKIP"

# Export variables for xargs
export EXE

# Function to run a single job (for xargs)
run_job() {
    set -e
    L=$1
    B=$2
    N_steps=$3
    gamma=$4
    alpha=$5
    mu=$6
    kill_radius=$7
    steps_to_record=$8
    recording_skip=$9
    echo "Starting: gamma=$gamma alpha=$alpha (L=$L, B=$B, N_steps=$N_steps)"
    $EXE "$L" "$B" "$N_steps" "$gamma" "$alpha" "$mu" "$kill_radius" "$steps_to_record" "$recording_skip"
    echo "Finished: gamma=$gamma alpha=$alpha"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 9 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"