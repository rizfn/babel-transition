#!/bin/bash

# Parameter grids
gammas=(-4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10)
alphas=(-4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10)
N=1000
B=4
NRounds=500
mus=(0.01)
generations=1000

# Number of parallel jobs
N_JOBS=$(($(nproc) - 4))

run_sim() {
    gamma=$1
    alpha=$2
    N=$3
    B=$4
    NRounds=$5
    mu=$6
    generations=$7
    ./src/understandabilityVsHammingSmall/top50 "$gamma" "$alpha" "$N" "$B" "$NRounds" "$mu" "$generations"
}

export -f run_sim

if command -v parallel > /dev/null; then
    parallel -j $N_JOBS run_sim ::: "${gammas[@]}" ::: "${alphas[@]}" ::: "$N" ::: "$B" ::: "$NRounds" ::: "${mus[@]}" ::: "$generations"
else
    echo "GNU Parallel is not installed. Please install it to run this script."
    exit 1
fi