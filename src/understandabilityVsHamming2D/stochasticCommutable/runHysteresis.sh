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

# Parameter ranges
ALPHA_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8)
MU_VALUES=(0.0001)

# Fixed parameters
L=256
B=16
GAMMA=1
STEPS=120000
STEPS_TO_RECORD=5000
RECORDING_SKIP=1000
EQUILIBRIUM_STEPS=120000  # Added equilibrium steps parameter
N_SIMS=2  # Number of simulation runs for each parameter combination

EXE=./hysteresis

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    echo "Make sure to compile hysteresis.cpp first: g++ -O3 hysteresis.cpp -o hysteresis"
    exit 1
fi

echo "Fixed parameters: L=$L, B=$B, gamma=$GAMMA, steps=$STEPS, steps_to_record=$STEPS_TO_RECORD, recording_skip=$RECORDING_SKIP, equilibrium_steps=$EQUILIBRIUM_STEPS"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Mu values: ${MU_VALUES[*]}"
echo "Number of simulations per parameter set: $N_SIMS"

# Build the job list - cycle through simulations first, then parameters
JOBLIST=()
for sim in $(seq 1 $N_SIMS); do
    for alpha in "${ALPHA_VALUES[@]}"; do
        for mu in "${MU_VALUES[@]}"; do
            JOBLIST+=("$L $B $STEPS $GAMMA $alpha $mu $STEPS_TO_RECORD $RECORDING_SKIP $EQUILIBRIUM_STEPS $sim")
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
    L=$1
    B=$2
    steps=$3
    gamma=$4
    alpha=$5
    mu=$6
    steps_to_record=$7
    recording_skip=$8
    equilibrium_steps=$9
    sim=${10}
    echo "Starting hysteresis sim $sim: L=$L B=$B steps=$steps gamma=$gamma alpha=$alpha mu=$mu steps_to_record=$steps_to_record recording_skip=$recording_skip equilibrium_steps=$equilibrium_steps"
    $EXE "$L" "$B" "$steps" "$gamma" "$alpha" "$mu" "$steps_to_record" "$recording_skip" "$equilibrium_steps"
    echo "Finished hysteresis sim $sim: alpha=$alpha mu=$mu"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 10 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All hysteresis jobs completed!"