#!/bin/bash
# filepath: /home/rizfn/github/babel-transition/src/paper_draft/2D/runPopulationTimeseries.sh

# Number of CPUs to leave free
CPUS_FREE=2

# Get total logical CPUs
TOTAL_CPUS=$(nproc)
MAX_PARALLEL=$((TOTAL_CPUS - CPUS_FREE))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    MAX_PARALLEL=1
fi

echo "Total CPUs: $TOTAL_CPUS, Running with max $MAX_PARALLEL parallel jobs (leaving $CPUS_FREE free)"

# Parameter ranges - alpha varies and gamma is fixed
ALPHA_VALUES=(0.4 0.8 1.2)
MU_VALUES=(0.00001 0.0001 0.001 0.005)

# Fixed parameters (adjusted for populationTimeseries)
L=256
B=16
GAMMA=1
STEPS=120000
STEPS_TO_RECORD=100000
RECORDING_SKIP=100

EXE=./populationTimeseries

# Make sure the executable exists
if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found or not executable."
    echo "Please compile with: g++ -O3 -std=c++17 -o populationTimeseries populationTimeseries.cpp"
    exit 1
fi

# Create output directory structure
for alpha in "${ALPHA_VALUES[@]}"; do
    for mu in "${MU_VALUES[@]}"; do
        mkdir -p "outputs/populationTimeseries/L_${L}_B_${B}"
    done
done

echo "Fixed parameters: L=$L, B=$B, gamma=$GAMMA, steps=$STEPS, steps_to_record=$STEPS_TO_RECORD, recording_skip=$RECORDING_SKIP"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Mu values: ${MU_VALUES[*]}"

# Build the job list
JOBLIST=()
for alpha in "${ALPHA_VALUES[@]}"; do
    for mu in "${MU_VALUES[@]}"; do
        JOBLIST+=("$L $B $STEPS $GAMMA $alpha $mu $STEPS_TO_RECORD $RECORDING_SKIP")
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
    
    # Create output directory if it doesn't exist
    output_dir="outputs/populationTimeseries/L_${L}_B_${B}"
    mkdir -p "$output_dir"
    
    echo "Starting: L=$L B=$B steps=$steps gamma=$gamma alpha=$alpha mu=$mu steps_to_record=$steps_to_record recording_skip=$recording_skip"
    
    # Run the simulation
    $EXE "$L" "$B" "$steps" "$gamma" "$alpha" "$mu" "$steps_to_record" "$recording_skip"
    
    echo "Finished: alpha=$alpha mu=$mu (output in $output_dir/g_${gamma}_a_${alpha}_mu_${mu}.tsv)"
}

export -f run_job

# Run jobs in parallel
printf "%s\n" "${JOBLIST[@]}" | xargs -n 8 -P "$MAX_PARALLEL" bash -c 'run_job "$@"' _

echo "All jobs completed!"
