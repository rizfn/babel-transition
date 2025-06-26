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
#include <unordered_map>
#include <map>

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

constexpr int DEFAULT_L = 256; // lattice size
constexpr int DEFAULT_B = 16;  // bitstring length
constexpr int DEFAULT_N_STEPS = 1000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr double DEFAULT_MU = 0.001;
constexpr int DEFAULT_KILL_RADIUS = 1;
constexpr int DEFAULT_STEPS_TO_RECORD = 40000;
constexpr int DEFAULT_RECORDING_SKIP = 50;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent(int B) : language(B, 0) {}
};

class UnionFind
{
public:
    UnionFind(int n) : parent(n), rank(n, 0)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int i)
    {
        if (parent[i] != i)
            parent[i] = find(parent[i]);
        return parent[i];
    }

    void union_set(int i, int j)
    {
        int ri = find(i), rj = find(j);
        if (ri != rj)
        {
            if (rank[ri] < rank[rj])
                parent[ri] = rj;
            else if (rank[ri] > rank[rj])
                parent[rj] = ri;
            else
            {
                parent[ri] = rj;
                ++rank[rj];
            }
        }
    }

private:
    std::vector<int> parent, rank;
};

// Convert bitstring to string for use as map key
std::string bitstring_to_string(const std::vector<int>& bitstring)
{
    std::string result;
    result.reserve(bitstring.size());
    for (int bit : bitstring)
        result += std::to_string(bit);
    return result;
}

// Get cluster sizes for all unique languages
std::map<std::string, std::vector<int>> get_language_cluster_sizes(
    const std::vector<std::vector<Agent>>& lattice, int L)
{
    // Map from language string to lattice positions with that language
    std::map<std::string, std::vector<int>> language_positions;
    
    // Collect positions for each unique language
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            std::string lang_str = bitstring_to_string(lattice[i][j].language);
            language_positions[lang_str].push_back(i * L + j);
        }
    }
    
    std::map<std::string, std::vector<int>> cluster_sizes_by_language;
    
    // For each unique language, find its clusters
    for (const auto& [lang_str, positions] : language_positions)
    {
        if (positions.empty()) continue;
        
        UnionFind uf(L * L);
        
        // Create a set of positions for fast lookup
        std::set<int> position_set(positions.begin(), positions.end());
        
        // Union adjacent positions with the same language
        for (int pos : positions)
        {
            int i = pos / L;
            int j = pos % L;
            
            // Check all 4 neighbors (with periodic boundary conditions)
            int neighbors[4][2] = {
                {(i - 1 + L) % L, j},
                {(i + 1) % L, j},
                {i, (j - 1 + L) % L},
                {i, (j + 1) % L}
            };
            
            for (int n = 0; n < 4; ++n)
            {
                int ni = neighbors[n][0];
                int nj = neighbors[n][1];
                int neighbor_pos = ni * L + nj;
                
                if (position_set.count(neighbor_pos))
                {
                    uf.union_set(pos, neighbor_pos);
                }
            }
        }
        
        // Count cluster sizes
        std::unordered_map<int, int> cluster_sizes;
        for (int pos : positions)
        {
            int root = uf.find(pos);
            ++cluster_sizes[root];
        }
        
        // Convert to vector
        std::vector<int> sizes;
        for (const auto& [root, size] : cluster_sizes)
            sizes.push_back(size);
        
        cluster_sizes_by_language[lang_str] = sizes;
    }
    
    return cluster_sizes_by_language;
}

int communicability(const std::vector<int> &a, const std::vector<int> &b)
{
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        count += (a[i] & b[i]);
    return count;
}

// Calculate mean-field distance (equivalent to hamming distance with mean field)
double mean_field_distance(const std::vector<int> &language, const std::vector<double> &mean_field)
{
    double distance = 0.0;
    for (size_t i = 0; i < language.size(); ++i)
    {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(language.size());
}

std::vector<int> mutate(const std::vector<int> &lang, double mu)
{
    std::vector<int> mutated = lang;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < lang.size(); ++i)
    {
        if (dis(gen) < mu)
            mutated[i] = 1 - mutated[i];
    }
    return mutated;
}

std::tuple<int, int> find_weakest_in_radius(
    const std::vector<std::vector<Agent>> &lattice,
    int cx, int cy, int radius, int L)
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
    int killRadius,
    int L,
    int B)
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

    // 3. Reproduction: kill-and-clone (no mutation here)
    std::vector<std::tuple<double, int, int>> agent_list;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            agent_list.emplace_back(lattice[i][j].fitness, i, j);
    std::shuffle(agent_list.begin(), agent_list.end(), gen);
    std::stable_sort(agent_list.begin(), agent_list.end(),
                     [](auto &a, auto &b)
                     {
                         return std::get<0>(a) > std::get<0>(b); // compare only fitness
                     });

    for (auto &row : lattice)
        for (auto &agent : row)
            agent.immune = false;

    for (auto &[fit, i, j] : agent_list)
    {
        if (lattice[i][j].immune)
            continue;
        auto [wi, wj] = find_weakest_in_radius(lattice, i, j, killRadius, L);
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
    int L,
    int B,
    int N_steps,
    double gamma,
    double alpha,
    double mu,
    int killRadius,
    int stepsToRecord,
    int recordingSkip,
    const std::string &output_path)
{
    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L, Agent(B)));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    std::ofstream fout(output_path);
    for (int step = 0; step <= N_steps; ++step)
    {
        update(lattice, gamma, alpha, mu, killRadius, L, B);

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << N_steps << "\r" << std::flush;

        // For the last stepsToRecord steps, record cluster sizes
        if ((step >= N_steps - stepsToRecord) && (step % recordingSkip == 0))
        {
            auto cluster_data = get_language_cluster_sizes(lattice, L);
            
            fout << step;
            
            // Write cluster sizes for each language
            for (const auto& [language, cluster_sizes] : cluster_data)
            {
                fout << "\t" << language << ":";
                for (size_t i = 0; i < cluster_sizes.size(); ++i)
                {
                    fout << (i == 0 ? "" : ",") << cluster_sizes[i];
                }
            }
            fout << "\n";
        }
    }
    fout.close();
}

int main(int argc, char *argv[])
{
    // Parse arguments in order: L, B, N_STEPS, gamma, alpha, mu, killRadius, stepsToRecord, recordingSkip
    int L = DEFAULT_L;
    int B = DEFAULT_B;
    int N_steps = DEFAULT_N_STEPS;
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    double mu = DEFAULT_MU;
    int killRadius = DEFAULT_KILL_RADIUS;
    int stepsToRecord = DEFAULT_STEPS_TO_RECORD;
    int recordingSkip = DEFAULT_RECORDING_SKIP;

    if (argc > 1) L = std::stoi(argv[1]);
    if (argc > 2) B = std::stoi(argv[2]);
    if (argc > 3) N_steps = std::stoi(argv[3]);
    if (argc > 4) gamma = std::stod(argv[4]);
    if (argc > 5) alpha = std::stod(argv[5]);
    if (argc > 6) mu = std::stod(argv[6]);
    if (argc > 7) killRadius = std::stoi(argv[7]);
    if (argc > 8) stepsToRecord = std::stoi(argv[8]);
    if (argc > 9) recordingSkip = std::stoi(argv[9]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/clusterTimeseries/constantGamma/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << "_K_" << killRadius << ".tsv";

    run(L, B, N_steps, gamma, alpha, mu, killRadius, stepsToRecord, recordingSkip, fname.str());

    return 0;
}