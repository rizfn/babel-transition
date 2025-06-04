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

constexpr int L = 100; // lattice size
constexpr int B = 16;  // bitstring length
constexpr int N_STEPS = 1000;
constexpr int KILL_RADIUS = 3;
constexpr double DEFAULT_GAMMA = 2.0;
constexpr double DEFAULT_ALPHA = 2.0;
constexpr double DEFAULT_GLOBAL_INTERACTION_RATIO = 2.0;
constexpr double DEFAULT_MU = 0.01;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent() : language(B, 0) {}
};

int manhattan(int x1, int y1, int x2, int y2)
{
    int dx = std::abs(x1 - x2);
    int dy = std::abs(y1 - y2);
    dx = std::min(dx, L - dx);
    dy = std::min(dy, L - dy);
    return dx + dy;
}

std::pair<int, int> random_neighbor(int x, int y)
{
    static const int dx[4] = {1, -1, 0, 0};
    static const int dy[4] = {0, 0, 1, -1};
    int dir = std::uniform_int_distribution<>(0, 3)(gen);
    int nx = (x + dx[dir] + L) % L;
    int ny = (y + dy[dir] + L) % L;
    return {nx, ny};
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
    // Randomly choose among weakest
    if (!candidates.empty())
    {
        int idx = std::uniform_int_distribution<>(0, candidates.size() - 1)(gen);
        return {candidates[idx].first, candidates[idx].second};
    }
    // fallback: return self
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

    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    std::uniform_int_distribution<> dis_L(0, L - 1);

    for (int step = 0; step < steps; ++step)
    {
        // 1. Fitness evaluation: reset fitness
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                lattice[i][j].fitness = 0.0;

        // 1a. Local interactions (with 4 neighbors, only simplicity matters)
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                Agent &a = lattice[i][j];
                // 4 neighbors: up, down, left, right (with periodic BCs)
                int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
                int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};
                for (int d = 0; d < 4; ++d)
                {
                    // Only simplicity (alpha term)
                    double fit = alpha * (double(sumBits(a.language)) / B);
                    a.fitness += fit;
                }
            }
        }

        // 1b. Global interactions (with random non-neighbours, only hamming/gamma matters)
        int n_global = static_cast<int>(4 * globalInteractionRatio);
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                Agent &a = lattice[i][j];
                for (int g = 0; g < n_global; ++g)
                {
                    // Pick a random non-neighbour site
                    int x2, y2;
                    do
                    {
                        x2 = dis_L(gen);
                        y2 = dis_L(gen);
                        // Exclude self and direct neighbours
                        bool is_neighbour = (x2 == i && y2 == j) ||
                                            (x2 == (i + 1) % L && y2 == j) ||
                                            (x2 == (i - 1 + L) % L && y2 == j) ||
                                            (x2 == i && y2 == (j + 1) % L) ||
                                            (x2 == i && y2 == (j - 1 + L) % L);
                        if (!is_neighbour)
                            break;
                    } while (true);
                    Agent &b = lattice[x2][y2];
                    int d_hamming = hamming(a.language, b.language);
                    double fit = gamma * (double(d_hamming) / B);
                    a.fitness += fit;
                }
            }
        }

        // Normalize fitness (each agent has 4 local + n_global global interactions)
        double norm = 4.0 + n_global;
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
            std::vector<int> new_lang1 = mutate(lattice[i][j].language, mu);
            std::vector<int> new_lang2 = mutate(lattice[i][j].language, mu);
            lattice[i][j].language = new_lang1;
            lattice[wi][wj].language = new_lang2;
            lattice[i][j].immune = true;
            lattice[wi][wj].immune = true;
        }

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;
    }

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Output final lattice to file
    std::ostringstream fname;
    fname << exeDir << "/outputs/lattice/neighbours/L_" << L << "_g_" << gamma << "_a_" << alpha << "_r_" << globalInteractionRatio << "_mu_" << mu << "_K_" << killRadius << ".tsv";
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