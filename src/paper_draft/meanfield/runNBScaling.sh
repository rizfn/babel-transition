#!/bin/bash

# Number of CPUs to leave free
CPUS_FREE=2

# Get total logical CPUs
TOTAL_CPUS=$(nproc)
MAX_PARALLEL=$((TOTAL_CPUS - CPUS_FREE))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    MAX_PARALLEL=1
fi

# Parameter ranges
GAMMA=1.0
ALPHA_VALUES=(0 0.2 0.4 0.6 0.8 1.0)
N_SIMS=4  # Number of simulations per parameter set

START_SIM_NO=16

# Simulation sets:
# 1. Default: N=65536, B=16
# 2. N=65536, B=32
# 3. N=65536, B=64
# 4. N=131072, B=16
# 5. N=32768,  B=16

SIM_SETS=(
    "65536 16"
    "65536 32"
    "65536 64"
    "131072 16"
    "32768 16"
)

BASE_B=16
BASE_MU=0.0001
GENERATIONS=10000
LAST_GENS_TO_RECORD=2000
RECORDING_SKIP=100

EXE=./NBScaling  # Path to your executable

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    exit 1
fi

# Build the job list
JOBLIST=()
for simset in "${SIM_SETS[@]}"; do
    set -- $simset
    N=$1
    B=$2
    # Scale mu according to B
    MU=$(awk "BEGIN {printf \"%.10g\", $BASE_MU * $BASE_B / $B}")
    for alpha in "${ALPHA_VALUES[@]}"; do
        for ((sim=0; sim<N_SIMS; sim++)); do
            simNo=$((START_SIM_NO + sim))
            JOBLIST+=("$GAMMA $alpha $N $B $MU $GENERATIONS $LAST_GENS_TO_RECORD $RECORDING_SKIP $simNo")
        done
    done
done

TOTAL_JOBS=${#JOBLIST[@]}

# Export variables for xargs
export EXE

# Function to run a single job (for xargs)
run_job() {
    set -e
    gamma=$1
    alpha=$2
    N=$3
    B=$4
    mu=$5
    generations=$6
    last_gens_to_record=$7
    recording_skip=$8
    simNo=$9
    echo "Starting: alpha=$alpha N=$N B=$B simNo=$simNo"
    $EXE "$gamma" "$alpha" "$N" "$B" "$mu" "$generations" "$last_gens_to_record" "$recording_skip" "$simNo"
    echo "Finished: alpha=$alpha N=$N B=$B simNo=$simNo"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 9 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"