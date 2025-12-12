#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <filesystem>

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
constexpr int DEFAULT_L = 256;
constexpr int DEFAULT_B = 16;
constexpr int DEFAULT_N_STEPS = 60000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr double DEFAULT_MU = 0.0001;
constexpr int DEFAULT_STEPS_TO_RECORD = 40000;
constexpr int DEFAULT_RECORDING_SKIP = 500;
constexpr int DEFAULT_MIN_CLUSTER_THRESHOLD = 2;

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

std::string bitstring_to_string(const std::vector<int>& bitstring)
{
    std::string result;
    result.reserve(bitstring.size());
    for (int bit : bitstring)
        result += std::to_string(bit);
    return result;
}

std::tuple<int, double, std::vector<int>> get_cluster_info(const std::vector<std::vector<std::vector<int>>>& lattice, int L, int minClusterThreshold)
{
    std::unordered_map<std::string, std::vector<int>> language_positions;

    // Collect positions for each unique language
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            std::string lang_str = bitstring_to_string(lattice[i][j]);
            language_positions[lang_str].push_back(i * L + j);
        }
    }

    int total_clusters_above_threshold = 0;
    int max_cluster_size = 0;
    std::vector<int> all_cluster_sizes;

    // For each unique language, find its clusters
    for (const auto& [lang_str, positions] : language_positions)
    {
        if (positions.empty()) continue;

        UnionFind uf(L * L);
        std::set<int> position_set(positions.begin(), positions.end());

        // Union adjacent positions with the same language
        for (int pos : positions)
        {
            int i = pos / L;
            int j = pos % L;

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

        for (const auto& [root, size] : cluster_sizes)
        {
            all_cluster_sizes.push_back(size);
            if (size >= minClusterThreshold)
            {
                ++total_clusters_above_threshold;
                max_cluster_size = std::max(max_cluster_size, size);
            }
        }
    }

    double largest_cluster_fraction = static_cast<double>(max_cluster_size) / static_cast<double>(L * L);
    return {total_clusters_above_threshold, largest_cluster_fraction, all_cluster_sizes};
}

int communicability(const std::vector<int> &a, const std::vector<int> &b, int B)
{
    int count = 0;
    for (int i = 0; i < B; ++i)
        count += (a[i] & b[i]);
    return count;
}

double mean_field_distance(const std::vector<int> &language, const std::vector<double> &mean_field, int B)
{
    double distance = 0.0;
    for (int i = 0; i < B; ++i)
    {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(B);
}

double calculate_fitness(const std::vector<std::vector<std::vector<int>>> &lattice, int i, int j,
                        const std::vector<double> &mean_field, double gamma, double alpha, int L, int B)
{
    const std::vector<int> &language = lattice[i][j];
    double fitness = 0.0;

    // Global interaction with mean field
    fitness += gamma * mean_field_distance(language, mean_field, B);

    // Local interactions with neighbors
    int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
    int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};

    double local_fitness = 0.0;
    for (int d = 0; d < 4; ++d)
    {
        const std::vector<int> &neighbor = lattice[ni[d]][nj[d]];
        int comm = communicability(language, neighbor, B);
        local_fitness += (alpha / 4.0) * (static_cast<double>(comm) / static_cast<double>(B));
    }

    fitness += local_fitness;
    return fitness;
}

std::vector<double> calculate_mean_field(const std::vector<std::vector<std::vector<int>>> &lattice, int L, int B)
{
    std::vector<double> mean_field(B, 0.0);
    int total_agents = L * L;

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            for (int b = 0; b < B; ++b)
            {
                mean_field[b] += static_cast<double>(lattice[i][j][b]);
            }
        }
    }

    // Normalize to get probabilities
    for (int b = 0; b < B; ++b)
    {
        mean_field[b] /= static_cast<double>(total_agents);
    }

    return mean_field;
}

void update(std::vector<std::vector<std::vector<int>>> &lattice, std::vector<double> &mean_field, double gamma, double alpha, double mu, int L, int B)
{
    std::uniform_int_distribution<> site_dist(0, L - 1);
    std::uniform_int_distribution<> neighbor_dist(0, 3);
    std::uniform_real_distribution<> mutation_dist(0.0, 1.0);

    int total_substeps = L * L;
    int total_agents = L * L;

    for (int substep = 0; substep < total_substeps; ++substep)
    {
        // 1. Choose a random site
        int i = site_dist(gen);
        int j = site_dist(gen);

        // 2. Choose a random neighbor
        int neighbor_idx = neighbor_dist(gen);
        int ni[4] = {(i + 1) % L, (i - 1 + L) % L, i, i};
        int nj[4] = {j, j, (j + 1) % L, (j - 1 + L) % L};
        int ni_chosen = ni[neighbor_idx];
        int nj_chosen = nj[neighbor_idx];

        // Save initial values of both sites
        std::vector<int> site_old = lattice[i][j];
        std::vector<int> neighbor_old = lattice[ni_chosen][nj_chosen];

        // 3. Calculate fitness of both sites
        double fitness_site = calculate_fitness(lattice, i, j, mean_field, gamma, alpha, L, B);
        double fitness_neighbor = calculate_fitness(lattice, ni_chosen, nj_chosen, mean_field, gamma, alpha, L, B);

        // 4. If fitnesses are unequal, stronger invades weaker
        if (fitness_site > fitness_neighbor)
        {
            lattice[ni_chosen][nj_chosen] = lattice[i][j];
        }
        else if (fitness_neighbor > fitness_site)
        {
            lattice[i][j] = lattice[ni_chosen][nj_chosen];
        }

        // 5. Attempt to mutate both sites
        for (int b = 0; b < B; ++b)
        {
            if (mutation_dist(gen) < mu)
                lattice[i][j][b] = 1 - lattice[i][j][b];
            if (mutation_dist(gen) < mu)
                lattice[ni_chosen][nj_chosen][b] = 1 - lattice[ni_chosen][nj_chosen][b];
        }

        // 6. Update mean field: subtract old contributions, add new contributions
        for (int b = 0; b < B; ++b)
        {
            mean_field[b] -= static_cast<double>(site_old[b]) / total_agents;
            mean_field[b] += static_cast<double>(lattice[i][j][b]) / total_agents;
            mean_field[b] -= static_cast<double>(neighbor_old[b]) / total_agents;
            mean_field[b] += static_cast<double>(lattice[ni_chosen][nj_chosen][b]) / total_agents;
        }
    }
}

void run(int L, int B, double gamma, double alpha, double mu, int steps, int steps_to_record, int recording_skip, int minClusterThreshold, std::ofstream &stats_file, std::ofstream &csd_file)
{
    // Lattice initialization: all agents identical with all zeros
    std::vector<std::vector<std::vector<int>>> lattice(L, std::vector<std::vector<int>>(L, std::vector<int>(B, 0)));

    // Calculate mean field once at the beginning
    std::vector<double> mean_field = calculate_mean_field(lattice, L, B);

    for (int step = 0; step < steps; ++step)
    {
        update(lattice, mean_field, gamma, alpha, mu, L, B);

        if (step % 10000 == 0)
            std::cout << "Step " << step << "/" << steps << "\n" << std::flush;

        // For the last steps_to_record steps, record cluster statistics
        if ((step >= steps - steps_to_record) && (step % recording_skip == 0))
        {
            auto [num_clusters, largest_cluster_fraction, cluster_sizes] = get_cluster_info(lattice, L, minClusterThreshold);

            // Write cluster statistics
            stats_file << step << "\t" << num_clusters << "\t" << std::fixed << std::setprecision(6)
                 << largest_cluster_fraction << "\n";

            // Write cluster size distribution
            csd_file << step << "\t";
            for (size_t i = 0; i < cluster_sizes.size(); ++i)
            {
                csd_file << cluster_sizes[i];
                if (i < cluster_sizes.size() - 1)
                    csd_file << ",";
            }
            csd_file << "\n";
        }
    }
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
    int minClusterThreshold = DEFAULT_MIN_CLUSTER_THRESHOLD;
    int simNumber = 0;

    if (argc > 1) L = std::stoi(argv[1]);
    if (argc > 2) B = std::stoi(argv[2]);
    if (argc > 3) steps = std::stoi(argv[3]);
    if (argc > 4) gamma = std::stod(argv[4]);
    if (argc > 5) alpha = std::stod(argv[5]);
    if (argc > 6) mu = std::stod(argv[6]);
    if (argc > 7) steps_to_record = std::stoi(argv[7]);
    if (argc > 8) recording_skip = std::stoi(argv[8]);
    if (argc > 9) minClusterThreshold = std::stoi(argv[9]);
    if (argc > 10) simNumber = std::stoi(argv[10]);

    std::ostringstream filePathStream;
    filePathStream << "/nbi/nbicmplx/cell/rpw391/babel2D/orderParamL/outputs/L_" << L << "_B_" << B << "/g_" << gamma << "_a_" << alpha << "_mu_" << mu << "_" << simNumber << ".tsv";
    std::string filePath = filePathStream.str();

    std::ostringstream csdPathStream;
    csdPathStream << "/nbi/nbicmplx/cell/rpw391/babel2D/orderParamL/outputs/CSD_L_" << L << "_B_" << B << "/g_" << gamma << "_a_" << alpha << "_mu_" << mu << "_" << simNumber << ".tsv";
    std::string csdPath = csdPathStream.str();

    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());
    std::filesystem::create_directories(std::filesystem::path(csdPath).parent_path());

    std::ofstream stats_file;
    stats_file.open(filePath);
    stats_file << "step\tnumber_of_clusters\tlargest_cluster_fraction\n";

    std::ofstream csd_file;
    csd_file.open(csdPath);

    run(L, B, gamma, alpha, mu, steps, steps_to_record, recording_skip, minClusterThreshold, stats_file, csd_file);

    stats_file.close();
    csd_file.close();

    return 0;
}