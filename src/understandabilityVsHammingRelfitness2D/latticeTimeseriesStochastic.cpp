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
constexpr int KILL_RADIUS = 1;
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
    double mu,
    int killRadius)
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

            // 2a. Global interaction with mean field
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field);
            agent.fitness += global_fitness;

            // 2b. Local interactions with neighbors
            double local_fitness = 0.0;
            int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
            int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};

            for (int d = 0; d < 4; ++d)
            {
                Agent &neighbor = lattice[ni[d]][nj[d]];
                int comm = communicability(agent.language, neighbor.language);
                local_fitness += (alpha / 4.0) * (static_cast<double>(comm) / static_cast<double>(B));
            }

            agent.fitness += local_fitness;
        }
    }

    // 3. Stochastic reproduction
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);

    // Reset immunity
    for (auto &row : lattice)
        for (auto &agent : row)
            agent.immune = false;

    // Create indices for efficient random sampling without erasing
    std::vector<int> site_indices(L * L);
    std::iota(site_indices.begin(), site_indices.end(), 0);
    int active_sites = L * L;

    int reproductions_performed = 0;
    int target_reproductions = (L * L) / 2;

    while (reproductions_performed < target_reproductions && active_sites > 0)
    {
        // Choose a random active site
        int idx = std::uniform_int_distribution<>(0, active_sites - 1)(gen);
        int site_linear = site_indices[idx];
        int site_i = site_linear / L;
        int site_j = site_linear % L;

        // Get neighbor coordinates
        int ni[4] = {(site_i + 1) % L, (site_i - 1 + L) % L, site_i, site_i};
        int nj[4] = {site_j, site_j, (site_j + 1) % L, (site_j - 1 + L) % L};

        // Find available neighbors
        std::vector<int> available_neighbors;
        for (int d = 0; d < 4; ++d) {
            if (!lattice[ni[d]][nj[d]].immune) {
                available_neighbors.push_back(d);
            }
        }

        if (available_neighbors.empty()) {
            // No available neighbors, remove this site from active list
            std::swap(site_indices[idx], site_indices[active_sites - 1]);
            active_sites--;
            continue;
        }

        // Randomly pick one of the available neighbors
        int chosen_neighbor_idx = available_neighbors[std::uniform_int_distribution<>(0, available_neighbors.size() - 1)(gen)];
        int neighbor_i = ni[chosen_neighbor_idx];
        int neighbor_j = nj[chosen_neighbor_idx];

        // Calculate fitnesses
        double site_fitness = lattice[site_i][site_j].fitness;
        double neighbor_fitness = lattice[neighbor_i][neighbor_j].fitness;
        double total_fitness = site_fitness + neighbor_fitness;

        // Avoid division by zero
        if (total_fitness <= 0) {
            std::swap(site_indices[idx], site_indices[active_sites - 1]);
            active_sites--;
            continue;
        }

        // Determine who reproduces based on relative fitness
        double site_prob = site_fitness / total_fitness;
        
        if (prob_dist(gen) < site_prob) {
            // Site reproduces into neighbor
            lattice[neighbor_i][neighbor_j].language = lattice[site_i][site_j].language;
            lattice[neighbor_i][neighbor_j].immune = true;
        } else {
            // Neighbor reproduces into site
            lattice[site_i][site_j].language = lattice[neighbor_i][neighbor_j].language;
            lattice[site_i][site_j].immune = true;
        }

        // Remove the used site from active list
        std::swap(site_indices[idx], site_indices[active_sites - 1]);
        active_sites--;
        reproductions_performed++;
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
    int killRadius = KILL_RADIUS;
    int steps = N_STEPS;
    if (argc > 1)
        gamma = std::stod(argv[1]);
    if (argc > 2)
        alpha = std::stod(argv[2]);
    if (argc > 3)
        mu = std::stod(argv[3]);
    if (argc > 4)
        killRadius = std::stoi(argv[4]);
    if (argc > 5)
        steps = std::stoi(argv[5]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/latticeTimeseries/rasterscanMu/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << "_K_" << killRadius << ".tsv";

    run(gamma, alpha, mu, killRadius, steps, fname.str());

    return 0;
}