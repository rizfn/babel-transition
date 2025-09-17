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

# Parameter ranges - now gamma and alpha vary
GAMMA_VALUES=(0 0.2 0.4 0.6 0.8 1.0)
ALPHA_VALUES=(0 0.2 0.4 0.6 0.8 1.0)

# Number of simulations per parameter set
N_SIMS=4

# Fixed parameters
L=256
B=16
MU=0.0001
STEPS=60000
STEPS_TO_RECORD=40000
RECORDING_SKIP=500

EXE=./largestClusterSizeTimeseries

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

echo "Fixed parameters: L=$L, B=$B, mu=$MU, steps=$STEPS, steps_to_record=$STEPS_TO_RECORD, recording_skip=$RECORDING_SKIP"
echo "Gamma values: ${GAMMA_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Number of simulations per parameter set: $N_SIMS"

# Build the job list - run one complete sweep at a time
JOBLIST=()
for sim in $(seq 1 $N_SIMS); do
    for gamma in "${GAMMA_VALUES[@]}"; do
        for alpha in "${ALPHA_VALUES[@]}"; do
            JOBLIST+=("$L $B $STEPS $gamma $alpha $MU $STEPS_TO_RECORD $RECORDING_SKIP $sim")
        done
    done
done

TOTAL_JOBS=${#JOBLIST[@]}
echo "Total jobs to run: $TOTAL_JOBS (${#GAMMA_VALUES[@]} gamma × ${#ALPHA_VALUES[@]} alpha × $N_SIMS sims)"

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
    sim=$9
    echo "Starting sim $sim: L=$L B=$B steps=$steps gamma=$gamma alpha=$alpha mu=$mu steps_to_record=$steps_to_record recording_skip=$recording_skip"
    $EXE "$L" "$B" "$steps" "$gamma" "$alpha" "$mu" "$steps_to_record" "$recording_skip"
    echo "Finished sim $sim: gamma=$gamma alpha=$alpha"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 9 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"