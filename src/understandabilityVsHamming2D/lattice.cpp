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
constexpr int B = 16; // bitstring length
constexpr int N_STEPS = 1000;
constexpr int KILL_RADIUS = 3;
constexpr double DEFAULT_GAMMA = 1.0;
constexpr double DEFAULT_ALPHA = 2.0;
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

int communicability(const std::vector<int> &a, const std::vector<int> &b)
{
    int count = 0;
    for (int i = 0; i < B; ++i)
        count += (a[i] & b[i]);
    return count;
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

    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    for (int step = 0; step < steps; ++step)
    {
        // 1. Fitness evaluation: each agent plays with all 4 neighbors
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                Agent &a = lattice[i][j];
                a.fitness = 0.0;
                // 4 neighbors: up, down, left, right (with periodic BCs)
                int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
                int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};
                for (int d = 0; d < 4; ++d)
                {
                    Agent &b = lattice[ni[d]][nj[d]];
                    int d_hamming = hamming(a.language, b.language);
                    int d_comm = communicability(a.language, b.language);

                    double fit = alpha * (double(d_comm) / B) + gamma * (double(d_hamming) / B);
                    a.fitness += fit;
                }
                a.fitness /= 4.0; // average over 4 neighbors
            }
        }

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
    fname << exeDir << "/outputs/lattice/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << "_K_" << killRadius << ".tsv";
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