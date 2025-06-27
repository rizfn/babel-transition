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
constexpr double DEFAULT_MU = 0.001;
constexpr int STEPS_TO_RECORD = 40000;
constexpr int RECORDING_SKIP = 50;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;

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

    // 3. Global reproduction: replace weakest with strongest
    std::vector<std::tuple<double, int, int>> agent_list;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            agent_list.emplace_back(lattice[i][j].fitness, i, j);

    // RANDOMIZE order before sorting to eliminate positional bias
    std::shuffle(agent_list.begin(), agent_list.end(), gen);

    // Sort by fitness (highest first)
    std::sort(agent_list.begin(), agent_list.end(),
              [](const auto &a, const auto &b)
              {
                  return std::get<0>(a) > std::get<0>(b);
              });

    // Replace weakest half with strongest half
    int half_size = agent_list.size() / 2;
    for (int k = 0; k < half_size; ++k)
    {
        // Get strongest agent (from first half)
        auto [strong_fit, strong_i, strong_j] = agent_list[k];
        // Get weakest agent (from second half)
        auto [weak_fit, weak_i, weak_j] = agent_list[agent_list.size() - 1 - k];

        // Replace weakest with clone of strongest
        lattice[weak_i][weak_j].language = lattice[strong_i][strong_j].language;
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
    fname << exeDir << "/outputs/latticeTimeseries/globalKRaster/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << ".tsv";

    run(gamma, alpha, mu, steps, fname.str());

    return 0;
}