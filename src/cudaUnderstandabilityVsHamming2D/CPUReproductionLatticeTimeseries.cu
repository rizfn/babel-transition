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

static auto _ = []()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

std::random_device rd;
std::mt19937 gen(rd());

constexpr int L = 256; // lattice size
constexpr int B = 16;  // bitstring length
constexpr int N_STEPS = 1000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr int KILL_RADIUS = 10;
constexpr double DEFAULT_MU = 0.001;
constexpr int STEPS_TO_RECORD = 1000;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent() : language(B, 0) {}
};

std::vector<int> mutate(const std::vector<int> &lang, double mu)
{
    std::vector<int> mutated = lang;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < B; ++i)
    {
        if (dis(gen) < mu)
            mutated[i] = 1 - mutated[i];
    }
    return mutated;
}

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

__device__ int communicability_device(const int* d_lang, int offsetA, int offsetB)
{
    int comm = 0;
    for (int i = 0; i < B; i++)
        comm += (d_lang[offsetA + i] & d_lang[offsetB + i]);
    return comm;
}

__device__ int hamming_device(const int* d_lang, int offsetA, int offsetB)
{
    int dist = 0;
    for (int i = 0; i < B; i++)
        dist += (d_lang[offsetA + i] != d_lang[offsetB + i]);
    return dist;
}

__global__ void fitness_kernel(
    double *d_fitness,
    const int *d_lang,
    double alpha,
    double gamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= L || y >= L) return;

    int idx = x * L + y;
    int offsetA = idx * B;
    double local_fitness = 0.0;

    for (int x2 = 0; x2 < L; x2++)
    {
        for (int y2 = 0; y2 < L; y2++)
        {
            if (x2 == x && y2 == y) continue;

            int idx2 = x2 * L + y2;
            int offsetB = idx2 * B;

            if (is_neighbour_device(x, y, x2, y2))
            {
                // Understandability: sum of (bitA & bitB), normalized by B
                int comm = communicability_device(d_lang, offsetA, offsetB);
                local_fitness += (alpha / 4.0) * (double(comm) / double(B));
            }
            else
            {
                // Hamming distance: count of mismatching bits, normalized by B
                int dist = hamming_device(d_lang, offsetA, offsetB);
                local_fitness += (gamma / double(L * L - 5)) * (double(dist) / double(B));
            }
        }
    }

    d_fitness[idx] = local_fitness;
}

// Copy from host lattice to device, run kernel, copy back fitness
void gpu_compute_fitness(std::vector<std::vector<Agent>> &lattice,
                         double alpha, double gamma)
{
    // Flatten language array if needed (not used here for direct fitness calculation),
    // but we keep placeholders to align with original structure.
    // We'll just rely on thread-level logic for neighbor detection.

    size_t fitness_size = L * L * sizeof(double);
    double *d_fitness = nullptr;
    cudaMalloc(&d_fitness, fitness_size);
    cudaMemset(d_fitness, 0, fitness_size);

    // We don't need to copy 'language' to the device for the current approach
    // The logic is in the kernel with direct neighbor detection, but if we do
    // want to use the language vectors for some reason, we'd flatten them:
    size_t lang_size = L * L * B * sizeof(int);
    int *d_lang = nullptr;
    cudaMalloc(&d_lang, lang_size);
    std::vector<int> host_lang(L * L * B, 0);
    // Optionally pack the language data
    for(int i=0; i<L; ++i)
    {
        for(int j=0; j<L; ++j)
        {
            for(int b=0; b<B; ++b)
            {
                host_lang[(i * L + j) * B + b] = lattice[i][j].language[b];
            }
        }
    }
    cudaMemcpy(d_lang, host_lang.data(), lang_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
                 (L + blockDim.y - 1) / blockDim.y);

    fitness_kernel<<<gridDim, blockDim>>>(d_fitness, d_lang, alpha, gamma);
    cudaDeviceSynchronize();

    std::vector<double> host_fitness(L * L, 0.0);
    cudaMemcpy(host_fitness.data(), d_fitness, fitness_size, cudaMemcpyDeviceToHost);

    // Assign to lattice
    for(int i=0; i<L; ++i)
    {
        for(int j=0; j<L; ++j)
        {
            lattice[i][j].fitness = host_fitness[i * L + j];
        }
    }

    cudaFree(d_fitness);
    cudaFree(d_lang);
}

std::tuple<int, int> find_weakest_in_radius(
    const std::vector<std::vector<Agent>> &lattice,
    int cx, int cy, int radius)
{
    double min_fitness = std::numeric_limits<double>::infinity();
    std::vector<std::pair<int, int>> weakest_sites;
    for (int dx = -radius; dx <= radius; ++dx)
    {
        for (int dy = -radius; dy <= radius; ++dy)
        {
            if (std::abs(dx) + std::abs(dy) > radius)
                continue;
            int nx = (cx + dx + L) % L;
            int ny = (cy + dy + L) % L;
            if (!lattice[nx][ny].immune)
            {
                if (lattice[nx][ny].fitness < min_fitness)
                {
                    min_fitness = lattice[nx][ny].fitness;
                }
            }
        }
    }
    for (int dx = -radius; dx <= radius; ++dx)
    {
        for (int dy = -radius; dy <= radius; ++dy)
        {
            if (std::abs(dx) + std::abs(dy) > radius)
                continue;
            int nx = (cx + dx + L) % L;
            int ny = (cy + dy + L) % L;
            if (!lattice[nx][ny].immune && lattice[nx][ny].fitness == min_fitness)
            {
                weakest_sites.emplace_back(nx, ny);
            }
        }
    }
    if (!weakest_sites.empty())
    {
        int idx = std::uniform_int_distribution<>(0, weakest_sites.size() - 1)(gen);
        return {weakest_sites[idx].first, weakest_sites[idx].second};
    }
    return {cx, cy};
}

void update(
    std::vector<std::vector<Agent>> &lattice,
    double gamma,
    double alpha,
    double mu,
    int killRadius)
{
    // Compute fitness in parallel on GPU
    gpu_compute_fitness(lattice, alpha, gamma);

    // Reproduction (CPU)
    std::vector<std::tuple<double, int, int>> agent_list;
    agent_list.reserve(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            lattice[i][j].immune = false;
            agent_list.emplace_back(lattice[i][j].fitness, i, j);
        }
    }
    std::shuffle(agent_list.begin(), agent_list.end(), gen);
    std::stable_sort(agent_list.begin(), agent_list.end(),
                     [](auto &a, auto &b)
                     {
                         return std::get<0>(a) > std::get<0>(b);
                     });

    for (auto &[fit, x, y] : agent_list)
    {
        if (lattice[x][y].immune)
            continue;
        auto [wi, wj] = find_weakest_in_radius(lattice, x, y, killRadius);
        lattice[wi][wj].language = lattice[x][y].language;
        lattice[x][y].immune = true;
        lattice[wi][wj].immune = true;
    }

    // Mutation (CPU)
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int b = 0; b < B; ++b)
            {
                if (dis(gen) < mu)
                    lattice[i][j].language[b] = 1 - lattice[i][j].language[b];
            }
        }
    }
}

void run(
    double gamma,
    double alpha,
    double mu,
    int killRadius,
    int steps,
    const std::string &output_path)
{
    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    std::ofstream fout(output_path);
    for (int step = 0; step < steps; ++step)
    {
        update(lattice, gamma, alpha, mu, killRadius);

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;

        if (step >= steps - STEPS_TO_RECORD)
        {
            fout << step << "\t";
            for (int i = 0; i < L; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    for (int b = 0; b < B; ++b)
                        fout << lattice[i][j].language[b];
                    if (j < L - 1)
                        fout << ",";
                }
                if (i < L - 1)
                    fout << ";";
            }
            fout << "\n";
        }
    }
    fout.close();
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
    fname << exeDir << "/outputs/latticeTimeseriesCPURepro/L_" << L
          << "_g_" << gamma
          << "_a_" << alpha
          << "_B_" << B
          << "_mu_" << mu
          << "_K_" << killRadius
          << ".tsv";

    run(gamma, alpha, mu, killRadius, steps, fname.str());
    return 0;
}