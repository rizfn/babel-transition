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
#include <array>

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

constexpr int L = 100; // lattice size
constexpr int B = 16;  // bitstring length
constexpr int N_STEPS = 1000;
constexpr int KILL_RADIUS = 3;
constexpr double DEFAULT_GAMMA = 1.0;
constexpr double DEFAULT_ALPHA = 1.0;
constexpr double DEFAULT_GLOBAL_INTERACTION_RATIO = 0.0;
constexpr double DEFAULT_MU = 0.01;
constexpr int DEFAULT_MAX_DEPTH = 10;

// --- Genetic code tracking ---
using GeneticCode = std::array<long long, DEFAULT_MAX_DEPTH>;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round
    long long id = -1;   // unique id for this agent
    GeneticCode genetic_code;

    Agent() : language(B, 0), id(-1)
    {
        genetic_code.fill(-1);
    }
};

class GeneticTracker
{
private:
    long long next_id = 0;

public:
    GeneticCode generateChildCode(const GeneticCode &parent_code, long long parent_id)
    {
        GeneticCode child_code;
        child_code[0] = parent_id;
        for (int i = 1; i < DEFAULT_MAX_DEPTH; ++i)
            child_code[i] = parent_code[i - 1];
        return child_code;
    }

    int geneticDistance(const GeneticCode &code1, const GeneticCode &code2)
    {
        for (int i = 0; i < DEFAULT_MAX_DEPTH; ++i)
        {
            if (code1[i] == code2[i] && code1[i] != -1)
                return (i + 1);
        }
        return -1;
    }

    long long getNextId() { return next_id++; }
};

int manhattan(int x1, int y1, int x2, int y2)
{
    int dx = std::abs(x1 - x2);
    int dy = std::abs(y1 - y2);
    dx = std::min(dx, L - dx);
    dy = std::min(dy, L - dy);
    return dx + dy;
}

int hamming(const std::vector<int> &a, const std::vector<int> &b)
{
    int d = 0;
    for (int i = 0; i < B; ++i)
        d += (a[i] != b[i]);
    return d;
}

int sumBits(const std::vector<int> &bits)
{
    int sum = 0;
    for (int b : bits)
        sum += b;
    return sum;
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
    std::vector<std::pair<int, int>> candidates;
    for (int dx = -radius; dx <= radius; ++dx)
    {
        for (int dy = -radius; dy <= radius; ++dy)
        {
            if (std::abs(dx) + std::abs(dy) > radius)
                continue;
            int nx = (cx + dx + L) % L;
            int ny = (cy + dy + L) % L;
            if (lattice[nx][ny].immune)
                continue;
            if (lattice[nx][ny].fitness < min_fitness)
            {
                min_fitness = lattice[nx][ny].fitness;
                candidates.clear();
                candidates.emplace_back(nx, ny);
            }
            else if (lattice[nx][ny].fitness == min_fitness)
            {
                candidates.emplace_back(nx, ny);
            }
        }
    }
    if (!candidates.empty())
    {
        int idx = std::uniform_int_distribution<>(0, candidates.size() - 1)(gen);
        return {candidates[idx].first, candidates[idx].second};
    }
    return {cx, cy};
}

int main(int argc, char *argv[])
{
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    double globalInteractionRatio = DEFAULT_GLOBAL_INTERACTION_RATIO;
    double mu = DEFAULT_MU;
    int killRadius = KILL_RADIUS;
    int steps = N_STEPS;
    int max_depth = DEFAULT_MAX_DEPTH;
    if (argc > 1)
        gamma = std::stod(argv[1]);
    if (argc > 2)
        alpha = std::stod(argv[2]);
    if (argc > 3)
        globalInteractionRatio = std::stod(argv[3]);
    if (argc > 4)
        mu = std::stod(argv[4]);
    if (argc > 5)
        killRadius = std::stoi(argv[5]);
    if (argc > 6)
        steps = std::stoi(argv[6]);
    if (argc > 7)
        max_depth = std::stoi(argv[7]);

    GeneticTracker genetic_tracker;

    // Lattice initialization: all agents identical, assign unique ids and genetic codes
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L));
    long long agent_id = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
        {
            lattice[i][j].language = std::vector<int>(B, 0);
            lattice[i][j].id = agent_id;
            lattice[i][j].genetic_code.fill(-1);
            lattice[i][j].genetic_code[0] = agent_id;
            agent_id++;
        }

    std::uniform_int_distribution<> dis_L(0, L - 1);

    for (int step = 0; step < steps; ++step)
    {
        // 1. Fitness evaluation: reset fitness
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                lattice[i][j].fitness = 0.0;

        // 1a. Local interactions (with 4 neighbors, check relatedness)
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                Agent &a = lattice[i][j];
                int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
                int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};
                for (int d = 0; d < 4; ++d)
                {
                    Agent &b = lattice[ni[d]][nj[d]];
                    int d_genetic = genetic_tracker.geneticDistance(a.genetic_code, b.genetic_code);
                    int d_hamming = hamming(a.language, b.language);
                    if (d_genetic > 0)
                    {
                        // Related: fitness by simplicity (normalized)
                        double fit = alpha * (double(sumBits(a.language)) / B);
                        a.fitness += fit;
                    }
                    else
                    {
                        // Unrelated: fitness by hamming (normalized)
                        double fit = gamma * (double(d_hamming) / B);
                        a.fitness += fit;
                    }
                }
            }
        }

        // 1b. Global interactions (with random non-neighbours, check relatedness)
        int n_global = static_cast<int>(4 * globalInteractionRatio);
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                Agent &a = lattice[i][j];
                for (int g = 0; g < n_global; ++g)
                {
                    int x2, y2;
                    do
                    {
                        x2 = dis_L(gen);
                        y2 = dis_L(gen);
                        bool is_neighbour = (x2 == i && y2 == j) ||
                                            (x2 == (i + 1) % L && y2 == j) ||
                                            (x2 == (i - 1 + L) % L && y2 == j) ||
                                            (x2 == i && y2 == (j + 1) % L) ||
                                            (x2 == i && y2 == (j - 1 + L) % L);
                        if (!is_neighbour)
                            break;
                    } while (true);
                    Agent &b = lattice[x2][y2];
                    int d_genetic = genetic_tracker.geneticDistance(a.genetic_code, b.genetic_code);
                    int d_hamming = hamming(a.language, b.language);
                    if (d_genetic > 0)
                    {
                        // Related: fitness by simplicity (normalized)
                        double fit = alpha * (double(sumBits(a.language)) / B);
                        a.fitness += fit;
                    }
                    else
                    {
                        // Unrelated: fitness by hamming (normalized)
                        double fit = gamma * (double(d_hamming) / B);
                        a.fitness += fit;
                    }
                }
            }
        }

        // Normalize fitness (each agent has 4 local + n_global global interactions)
        double norm = 4.0 + static_cast<int>(4 * globalInteractionRatio);
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                lattice[i][j].fitness /= norm;

        // 2. Reproduction: kill-and-clone
        std::vector<std::tuple<double, int, int>> agent_list;
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                agent_list.emplace_back(lattice[i][j].fitness, i, j);
        std::sort(agent_list.rbegin(), agent_list.rend());

        for (auto &row : lattice)
            for (auto &agent : row)
                agent.immune = false;

        for (auto &[fit, i, j] : agent_list)
        {
            if (lattice[i][j].immune)
                continue;
            auto [wi, wj] = find_weakest_in_radius(lattice, i, j, killRadius);
            if (lattice[wi][wj].immune)
                continue;
            // Mutate and assign new ids/genetic codes
            long long new_id1 = genetic_tracker.getNextId();
            long long new_id2 = genetic_tracker.getNextId();
            std::vector<int> new_lang1 = mutate(lattice[i][j].language, mu);
            std::vector<int> new_lang2 = mutate(lattice[i][j].language, mu);

            GeneticCode new_code1 = genetic_tracker.generateChildCode(lattice[i][j].genetic_code, lattice[i][j].id);
            GeneticCode new_code2 = genetic_tracker.generateChildCode(lattice[i][j].genetic_code, lattice[i][j].id);

            lattice[i][j].language = new_lang1;
            lattice[i][j].id = new_id1;
            lattice[i][j].genetic_code = new_code1;

            lattice[wi][wj].language = new_lang2;
            lattice[wi][wj].id = new_id2;
            lattice[wi][wj].genetic_code = new_code2;

            lattice[i][j].immune = true;
            lattice[wi][wj].immune = true;
        }

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;
    }

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Output final lattice to file
    std::ostringstream fname;
    fname << exeDir << "/outputs/lattice/geneticDist/L_" << L << "_g_" << gamma << "_a_" << alpha
          << "_r_" << globalInteractionRatio << "_mu_" << mu << "_K_" << killRadius
          << "_d_" << max_depth << ".tsv";
    std::ofstream fout(fname.str());
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int b = 0; b < B; ++b)
                fout << lattice[i][j].language[b];
            if (j < L - 1)
                fout << "\t";
        }
        fout << "\n";
    }
    fout.close();
    return 0;
}