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

// Default constants
constexpr int DEFAULT_L = 512;
constexpr int DEFAULT_B = 16;
constexpr int DEFAULT_N_STEPS = 1000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr double DEFAULT_MU = 0.001;
constexpr int DEFAULT_STEPS_TO_RECORD = 40000;
constexpr int DEFAULT_RECORDING_SKIP = 50;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent(int B) : language(B, 0) {}
};

int communicability(const std::vector<int> &a, const std::vector<int> &b, int B)
{
    int count = 0;
    for (int i = 0; i < B; ++i)
        count += (a[i] & b[i]);
    return count;
}

// Calculate mean-field distance (equivalent to hamming distance with mean field)
double mean_field_distance(const std::vector<int> &language, const std::vector<double> &mean_field, int B)
{
    double distance = 0.0;
    for (int i = 0; i < B; ++i)
    {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(B);
}

std::tuple<int, int> find_weakest_neighbor(
    const std::vector<std::vector<Agent>> &lattice, int cx, int cy, int L)
{
    double min_fitness = std::numeric_limits<double>::infinity();
    std::vector<std::pair<int, int>> weakest_neighbors;
    
    // Check all 4 neighbors
    int ni[4] = {(cx + 1) % L, (cx - 1 + L) % L, cx, cx};
    int nj[4] = {cy, cy, (cy + 1) % L, (cy - 1 + L) % L};
    
    // First pass: find minimum fitness among non-immune neighbors
    for (int d = 0; d < 4; ++d)
    {
        int nx = ni[d];
        int ny = nj[d];
        if (!lattice[nx][ny].immune)
        {
            if (lattice[nx][ny].fitness < min_fitness)
            {
                min_fitness = lattice[nx][ny].fitness;
            }
        }
    }
    
    // Second pass: collect all non-immune neighbors with min_fitness
    for (int d = 0; d < 4; ++d)
    {
        int nx = ni[d];
        int ny = nj[d];
        if (!lattice[nx][ny].immune && lattice[nx][ny].fitness == min_fitness)
        {
            weakest_neighbors.emplace_back(nx, ny);
        }
    }
    
    if (!weakest_neighbors.empty())
    {
        int idx = std::uniform_int_distribution<>(0, weakest_neighbors.size() - 1)(gen);
        return {weakest_neighbors[idx].first, weakest_neighbors[idx].second};
    }
    
    // No valid neighbor found (all neighbors are immune or stronger)
    return {-1, -1};
}

void update(std::vector<std::vector<Agent>> &lattice, double gamma, double alpha, double mu, int L, int B)
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
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field, B);
            agent.fitness += global_fitness;
            
            // 2b. Local interactions with neighbors
            double local_fitness = 0.0;
            int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
            int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};
            
            for (int d = 0; d < 4; ++d)
            {
                Agent &neighbor = lattice[ni[d]][nj[d]];
                int comm = communicability(agent.language, neighbor.language, B);
                local_fitness += (alpha / 4.0) * (static_cast<double>(comm) / static_cast<double>(B));
            }
            
            agent.fitness += local_fitness;
        }
    }

    // 3. Reproduction: stochastic invasion trials
    // Create a list of all lattice positions
    std::vector<std::pair<int, int>> positions;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            positions.emplace_back(i, j);
    
    // Shuffle the positions randomly
    std::shuffle(positions.begin(), positions.end(), gen);

    // Reset immunity
    for (auto &row : lattice)
        for (auto &agent : row)
            agent.immune = false;

    // Perform L*L/2 invasion trials
    int trials = (L * L) / 2;
    for (int trial = 0; trial < trials && trial < positions.size(); ++trial)
    {
        auto [i, j] = positions[trial];
        
        // Skip if this agent is already immune
        if (lattice[i][j].immune)
            continue;
        
        // Find weakest neighbor
        auto [wi, wj] = find_weakest_neighbor(lattice, i, j, L);
        
        // If no valid neighbor found, skip this trial
        if (wi == -1 && wj == -1)
            continue;
        
        // Check if current agent is stronger than the weakest neighbor
        if (lattice[i][j].fitness > lattice[wi][wj].fitness)
        {
            // Invade: clone current agent into weakest neighbor position
            lattice[wi][wj].language = lattice[i][j].language;
            lattice[i][j].immune = true;
            lattice[wi][wj].immune = true;
        }
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

void run(int L, int B, double gamma, double alpha, double mu, int steps, int steps_to_record, int recording_skip, const std::string &output_path)
{
    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L, Agent(B)));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    std::ofstream fout(output_path);
    for (int step = 0; step < steps; ++step)
    {
        update(lattice, gamma, alpha, mu, L, B);

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;

        // For the last steps_to_record steps, record the lattice
        if ((step >= steps - steps_to_record) && (step % recording_skip == 0))
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
    int L = DEFAULT_L;
    int B = DEFAULT_B;
    int steps = DEFAULT_N_STEPS;
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    double mu = DEFAULT_MU;
    int steps_to_record = DEFAULT_STEPS_TO_RECORD;
    int recording_skip = DEFAULT_RECORDING_SKIP;
    
    if (argc > 1) L = std::stoi(argv[1]);
    if (argc > 2) B = std::stoi(argv[2]);
    if (argc > 3) steps = std::stoi(argv[3]);
    if (argc > 4) gamma = std::stod(argv[4]);
    if (argc > 5) alpha = std::stod(argv[5]);
    if (argc > 6) mu = std::stod(argv[6]);
    if (argc > 7) steps_to_record = std::stoi(argv[7]);
    if (argc > 8) recording_skip = std::stoi(argv[8]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/latticeTimeseries/rasterscanMu/L_" << L << "_B_" << B << "/g_" << gamma << "_a_" << alpha << "_mu_" << mu << ".tsv";

    run(L, B, gamma, alpha, mu, steps, steps_to_record, recording_skip, fname.str());

    return 0;
}