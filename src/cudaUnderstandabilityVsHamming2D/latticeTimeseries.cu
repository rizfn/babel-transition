// Note: DOESN'T WORK! there's a bug somewhere.

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <set>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <cuda_runtime.h>
#include <curand_kernel.h>

static auto _ = []()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

constexpr int L = 256; // lattice size
constexpr int B = 16;  // bitstring length
constexpr int N_STEPS = 1000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr int KILL_RADIUS = 1;
constexpr double DEFAULT_MU = 0.001;
constexpr int STEPS_TO_RECORD = 1000;
constexpr int BLOCK_LENGTH = 2;

struct Agent
{
    int language[B];
    double fitness = 0.0;
    bool immune = false;
};

__device__ bool is_neighbour_device(int x1, int y1, int x2, int y2)
{
    if ((x2 == (x1 + 1 + L) % L && y2 == y1) ||
        (x2 == (x1 - 1 + L) % L && y2 == y1) ||
        (x2 == x1 && y2 == (y1 + 1 + L) % L) ||
        (x2 == x1 && y2 == (y1 - 1 + L) % L))
    {
        return true;
    }
    return false;
}

__device__ int communicability_device(const int* lang1, const int* lang2)
{
    int comm = 0;
    for (int i = 0; i < B; i++)
        comm += (lang1[i] & lang2[i]);
    return comm;
}

__device__ int hamming_device(const int* lang1, const int* lang2)
{
    int dist = 0;
    for (int i = 0; i < B; i++)
        dist += (lang1[i] != lang2[i]);
    return dist;
}

__global__ void fitness_kernel(Agent *d_lattice, double alpha, double gamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;

    int idx = x * L + y;
    double local_fitness = 0.0;

    for (int x2 = 0; x2 < L; x2++)
    {
        for (int y2 = 0; y2 < L; y2++)
        {
            if (x2 == x && y2 == y) continue;

            int idx2 = x2 * L + y2;

            if (is_neighbour_device(x, y, x2, y2))
            {
                int comm = communicability_device(d_lattice[idx].language, d_lattice[idx2].language);
                local_fitness += (alpha / 4.0) * (double(comm) / double(B));
            }
            else
            {
                int dist = hamming_device(d_lattice[idx].language, d_lattice[idx2].language);
                local_fitness += (gamma / double(L * L - 5)) * (double(dist) / double(B));
            }
        }
    }

    d_lattice[idx].fitness = local_fitness;
}

__device__ int find_weakest_in_radius_device(Agent *d_lattice, int cx, int cy, int radius, curandState *localState)
{
    double min_fitness = 1e9;
    int weakest_sites[100]; // Maximum possible sites in radius
    int num_weakest = 0;
    
    for (int dx = -radius; dx <= radius; ++dx)
    {
        for (int dy = -radius; dy <= radius; ++dy)
        {
            if (abs(dx) + abs(dy) > radius) continue;
            
            int nx = (cx + dx + L) % L;
            int ny = (cy + dy + L) % L;
            int nidx = nx * L + ny;
            
            if (!d_lattice[nidx].immune)
            {
                if (d_lattice[nidx].fitness < min_fitness)
                {
                    min_fitness = d_lattice[nidx].fitness;
                    num_weakest = 0;
                    weakest_sites[num_weakest++] = nidx;
                }
                else if (d_lattice[nidx].fitness == min_fitness)
                {
                    weakest_sites[num_weakest++] = nidx;
                }
            }
        }
    }
    
    if (num_weakest > 0)
    {
        int chosen = curand(localState) % num_weakest;
        return weakest_sites[chosen];
    }
    return cx * L + cy; // fallback
}

__global__ void initCurand(curandState *state, unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;
    
    int index = x * L + y;
    curand_init(seed, index, 0, &state[index]);
}


__global__ void reproduce_kernel(Agent *d_lattice, curandState *state, int killRadius, int sector, int attempt)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    
    // Check if this block belongs to the current sector (A=0, B=1, C=2, D=3)
    int block_sector = ((block_x % 2) * 2 + (block_y % 2));
    if (block_sector != sector) return;
    
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    int x = block_x * BLOCK_LENGTH + local_x;
    int y = block_y * BLOCK_LENGTH + local_y;
    
    if (x >= L || y >= L) return;
    
    int idx = x * L + y;
    curandState localState = state[idx];
    
    // For each block, we want BLOCK_LENGTH*BLOCK_LENGTH/2 attempts total
    // This attempt corresponds to one of those attempts
    int sites_per_block = BLOCK_LENGTH * BLOCK_LENGTH;
    int attempts_per_block = sites_per_block / 2;
    
    // Only process if this thread's attempt number matches the current attempt
    int thread_attempt = local_x * BLOCK_LENGTH + local_y;
    if (thread_attempt >= attempts_per_block) return;
    if (thread_attempt != attempt) return;
    
    // Find the fittest non-immune agent in this block
    double max_fitness = -1e9;
    int best_idx = -1;
    
    for (int bx = 0; bx < BLOCK_LENGTH; bx++)
    {
        for (int by = 0; by < BLOCK_LENGTH; by++)
        {
            int block_site_x = block_x * BLOCK_LENGTH + bx;
            int block_site_y = block_y * BLOCK_LENGTH + by;
            if (block_site_x >= L || block_site_y >= L) continue;
            
            int block_site_idx = block_site_x * L + block_site_y;
            
            if (!d_lattice[block_site_idx].immune && 
                d_lattice[block_site_idx].fitness > max_fitness)
            {
                max_fitness = d_lattice[block_site_idx].fitness;
                best_idx = block_site_idx;
            }
        }
    }
    
    if (best_idx != -1)
    {
        // Convert back to x,y coordinates
        int best_x = best_idx / L;
        int best_y = best_idx % L;
        
        // Find weakest in radius
        int weak_idx = find_weakest_in_radius_device(d_lattice, best_x, best_y, killRadius, &localState);
        
        // Copy language from best agent to weakest
        for (int b = 0; b < B; b++)
        {
            d_lattice[weak_idx].language[b] = d_lattice[best_idx].language[b];
        }
        
        // Set immunity
        d_lattice[best_idx].immune = true;
        d_lattice[weak_idx].immune = true;
    }
    
    state[idx] = localState;
}

__global__ void reset_immunity_kernel(Agent *d_lattice)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;
    
    int idx = x * L + y;
    d_lattice[idx].immune = false;
}

__global__ void mutate_kernel(Agent *d_lattice, curandState *state, double mu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;
    
    int idx = x * L + y;
    curandState localState = state[idx];
    
    for (int b = 0; b < B; b++)
    {
        if (curand_uniform(&localState) < mu)
        {
            d_lattice[idx].language[b] = 1 - d_lattice[idx].language[b];
        }
    }
    
    state[idx] = localState;
}


__global__ void record_lattice_kernel(Agent *d_lattice, int *d_recorded_data, int step_offset, int total_steps_to_record)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;
    
    int lattice_idx = x * L + y;
    int step_idx = step_offset;
    
    // Calculate the base index for this step and agent
    int base_idx = step_idx * L * L * B + lattice_idx * B;
    
    // Copy all bits of the language
    for (int b = 0; b < B; b++)
    {
        d_recorded_data[base_idx + b] = d_lattice[lattice_idx].language[b];
    }
}


void update_gpu(Agent *d_lattice, curandState *d_state, double gamma, double alpha, double mu, int killRadius)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (L + blockDim.y - 1) / blockDim.y);
    
    // Compute fitness
    fitness_kernel<<<gridDim, blockDim>>>(d_lattice, alpha, gamma);
    cudaDeviceSynchronize();
    
    // Reset immunity
    reset_immunity_kernel<<<gridDim, blockDim>>>(d_lattice);
    cudaDeviceSynchronize();
    
    // Reproduction - iterate over sectors A, B, C, D
    dim3 block_blockDim(BLOCK_LENGTH, BLOCK_LENGTH);
    dim3 block_gridDim(L / BLOCK_LENGTH, L / BLOCK_LENGTH);
    
    // Each block has BLOCK_LENGTH*BLOCK_LENGTH sites, so attempts_per_block = sites/2
    int sites_per_block = BLOCK_LENGTH * BLOCK_LENGTH;
    int attempts_per_block = sites_per_block / 2;
    
    // For each attempt within each block
    for (int attempt = 0; attempt < attempts_per_block; ++attempt)
    {
        // For each sector (A, B, C, D)
        for (int sector = 0; sector < 4; ++sector)
        {
            reproduce_kernel<<<block_gridDim, block_blockDim>>>(d_lattice, d_state, killRadius, sector, attempt);
            cudaDeviceSynchronize();
        }
    }
    
    // Mutation - completely parallel across all sites
    mutate_kernel<<<gridDim, blockDim>>>(d_lattice, d_state, mu);
    cudaDeviceSynchronize();
}

void run(double gamma, double alpha, double mu, int killRadius, int steps, const std::string &output_path)
{
    // Initialize host lattice
    std::vector<Agent> host_lattice(L * L);
    for (int i = 0; i < L * L; ++i)
    {
        for (int b = 0; b < B; ++b)
        {
            host_lattice[i].language[b] = 0;
        }
        host_lattice[i].fitness = 0.0;
        host_lattice[i].immune = false;
    }
    
    // Allocate GPU memory for simulation
    Agent *d_lattice;
    curandState *d_state;
    cudaMalloc(&d_lattice, L * L * sizeof(Agent));
    cudaMalloc(&d_state, L * L * sizeof(curandState));
    
    // Allocate GPU memory for recording data
    int *d_recorded_data;
    size_t recorded_data_size = STEPS_TO_RECORD * L * L * B * sizeof(int);
    cudaMalloc(&d_recorded_data, recorded_data_size);
    
    // Copy to GPU
    cudaMemcpy(d_lattice, host_lattice.data(), L * L * sizeof(Agent), cudaMemcpyHostToDevice);
    
    // Initialize random states
    dim3 blockDim(16, 16);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (L + blockDim.y - 1) / blockDim.y);
    initCurand<<<gridDim, blockDim>>>(d_state, time(0));
    cudaDeviceSynchronize();
    
    int record_step_counter = 0;
    
    for (int step = 0; step < steps; ++step)
    {
        update_gpu(d_lattice, d_state, gamma, alpha, mu, killRadius);
        
        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;
        
        if (step >= steps - STEPS_TO_RECORD)
        {
            // Record to GPU memory instead of copying to host
            record_lattice_kernel<<<gridDim, blockDim>>>(d_lattice, d_recorded_data, record_step_counter, STEPS_TO_RECORD);
            cudaDeviceSynchronize();
            record_step_counter++;
        }
    }
    
    // Copy all recorded data back to host at once
    std::vector<int> host_recorded_data(STEPS_TO_RECORD * L * L * B);
    cudaMemcpy(host_recorded_data.data(), d_recorded_data, recorded_data_size, cudaMemcpyDeviceToHost);
    
    // Write to file
    std::ofstream fout(output_path);
    for (int step_idx = 0; step_idx < STEPS_TO_RECORD; ++step_idx)
    {
        int actual_step = steps - STEPS_TO_RECORD + step_idx;
        fout << actual_step << "\t";
        
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                int lattice_idx = i * L + j;
                int base_idx = step_idx * L * L * B + lattice_idx * B;
                
                for (int b = 0; b < B; ++b)
                    fout << host_recorded_data[base_idx + b];
                if (j < L - 1)
                    fout << ",";
            }
            if (i < L - 1)
                fout << ";";
        }
        fout << "\n";
    }
    fout.close();
    
    cudaFree(d_lattice);
    cudaFree(d_state);
    cudaFree(d_recorded_data);
}


int main(int argc, char *argv[])
{
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    double mu = DEFAULT_MU;
    int killRadius = KILL_RADIUS;
    int steps = N_STEPS;

    if (argc > 1) gamma = std::stod(argv[1]);
    if (argc > 2) alpha = std::stod(argv[2]);
    if (argc > 3) mu = std::stod(argv[3]);
    if (argc > 4) killRadius = std::stoi(argv[4]);
    if (argc > 5) steps = std::stoi(argv[5]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/latticeTimeseries/L_" << L
          << "_g_" << gamma
          << "_a_" << alpha
          << "_B_" << B
          << "_mu_" << mu
          << "_K_" << killRadius
          << ".tsv";

    run(gamma, alpha, mu, killRadius, steps, fname.str());
    return 0;
}