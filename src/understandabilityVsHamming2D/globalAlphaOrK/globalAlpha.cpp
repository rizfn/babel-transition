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

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

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
constexpr int KILL_RADIUS = 1; // Fixed to 1
constexpr double DEFAULT_MU = 0.001;
constexpr int STEPS_TO_RECORD = 40000;
constexpr int RECORDING_SKIP = 50;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent() : language(B, 0) {}
};

int communicability(const std::vector<int> &a, const std::vector<int> &b)
{
    int count = 0;
    for (int i = 0; i < B; ++i)
        count += (a[i] & b[i]);
    return count;
}

// Calculate mean-field communicability (fraction of bits that can communicate with mean field)
double mean_field_communicability(const std::vector<int> &language, const std::vector<double> &mean_field)
{
    double comm = 0.0;
    for (int i = 0; i < B; ++i)
    {
        // Agent has bit i set AND mean field has probability > 0 for bit i
        if (language[i] == 1)
        {
            comm += mean_field[i];
        }
    }
    return comm / static_cast<double>(B);
}

// Calculate mean-field distance (equivalent to hamming distance with mean field)
double mean_field_distance(const std::vector<int> &language, const std::vector<double> &mean_field)
{
    double distance = 0.0;
    for (int i = 0; i < B; ++i)
    {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(B);
}

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

std::tuple<int, int> find_weakest_in_radius(
    const std::vector<std::vector<Agent>> &lattice,
    int cx, int cy, int radius)
{
    double min_fitness = std::numeric_limits<double>::infinity();
    std::vector<std::pair<int, int>> weakest_sites;
    // First pass: find minimum fitness among non-immune
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
    // Second pass: collect all non-immune sites with min_fitness
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
    // fallback: return self
    return {cx, cy};
}

void update(
    std::vector<std::vector<Agent>> &lattice,
    double gamma,
    double alpha,
    double mu)
{
    // 1. Calculate mean-field bitstring
    std::vector<double> mean_field(B, 0.0);
    int total_agents = L * L;

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int b = 0; b < B; ++b)
            {
                mean_field[b] += static_cast<double>(lattice[i][j].language[b]);
            }
        }
    }

    // Normalize to get probabilities
    for (int b = 0; b < B; ++b)
    {
        mean_field[b] /= static_cast<double>(total_agents);
    }

    // 2. Fitness evaluation: reset fitness and calculate new fitness
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            Agent &agent = lattice[i][j];
            agent.fitness = 0.0;

            // 2a. Global interaction with mean field (distance-based, want to be different)
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field);
            agent.fitness += global_fitness;

            // 2b. Global communicability with mean field (want to be communicable)
            double comm_fitness = alpha * mean_field_communicability(agent.language, mean_field);
            agent.fitness += comm_fitness;
        }
    }

    // 3. Reproduction: kill-and-clone (local reproduction with immunity)
    std::vector<std::tuple<double, int, int>> agent_list;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            agent_list.emplace_back(lattice[i][j].fitness, i, j);
    
    // RANDOMIZE order before sorting to eliminate positional bias
    std::shuffle(agent_list.begin(), agent_list.end(), gen);
    
    // Sort by fitness (highest first)
    std::stable_sort(agent_list.begin(), agent_list.end(),
                     [](auto &a, auto &b)
                     {
                         return std::get<0>(a) > std::get<0>(b); // compare only fitness
                     });

    // Reset immunity for this round
    for (auto &row : lattice)
        for (auto &agent : row)
            agent.immune = false;

    // Process agents in fitness order
    for (auto &[fit, i, j] : agent_list)
    {
        if (lattice[i][j].immune)
            continue;
        
        auto [wi, wj] = find_weakest_in_radius(lattice, i, j, KILL_RADIUS);
        
        // Perfect clone, no mutation
        lattice[wi][wj].language = lattice[i][j].language;
        lattice[i][j].immune = true;
        lattice[wi][wj].immune = true;
    }

    // 4. Mutation: after all reproduction, mutate every cell in-place
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
        update(lattice, gamma, alpha, mu);

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;

        // For the last STEPS_TO_RECORD steps, record the lattice
        if ((step >= steps - STEPS_TO_RECORD) && (step % RECORDING_SKIP == 0))
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
    int steps = N_STEPS;
    if (argc > 1)
        gamma = std::stod(argv[1]);
    if (argc > 2)
        alpha = std::stod(argv[2]);
    if (argc > 3)
        mu = std::stod(argv[3]);
    if (argc > 4)
        steps = std::stoi(argv[4]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/latticeTimeseries/globalAlpha/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << ".tsv";

    run(gamma, alpha, mu, steps, fname.str());

    return 0;
}