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
#include <limits>
#include <bitset>

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
constexpr int DEFAULT_N_STEPS = 120000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1;
constexpr double DEFAULT_MU = 0.001;
constexpr int DEFAULT_STEPS_TO_RECORD = 110000;
constexpr int DEFAULT_RECORDING_SKIP = 500;
constexpr int DEFAULT_MIN_CLUSTER_SIZE = 10;

struct Agent
{
    std::vector<int> language;
    std::string parent_language; // Stores parent language as bitstring
    double fitness = 0.0;
    bool immune = false; // immune from elimination this round

    Agent(int B) : language(B, 0), parent_language("") {}
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

// Convert bitstring to integer for compact storage
uint64_t bitstring_to_int(const std::vector<int>& bitstring)
{
    uint64_t result = 0;
    for (size_t i = 0; i < bitstring.size(); ++i) {
        if (bitstring[i]) {
            result |= (1ULL << i);
        }
    }
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
            // Get current language of the invaded cell before replacing
            std::string invaded_language = bitstring_to_string(lattice[wi][wj].language);
            std::string invader_language = bitstring_to_string(lattice[i][j].language);
            
            // Invade: clone current agent into weakest neighbor position
            lattice[wi][wj].language = lattice[i][j].language;
            // Set parent language to the invader's language
            lattice[wi][wj].parent_language = invader_language;
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
            // Store original language before mutation
            std::string original_language = bitstring_to_string(lattice[i][j].language);
            bool mutated = false;
            
            for (int b = 0; b < B; ++b)
            {
                if (dis(gen) < mu) {
                    // Record that we're mutating this language
                    if (!mutated) {
                        lattice[i][j].parent_language = original_language;
                        mutated = true;
                    }
                    lattice[i][j].language[b] = 1 - lattice[i][j].language[b];
                }
            }
        }
    }
}

// Structure to represent a language node in the phylogenetic tree
struct LanguageNode {
    std::string bitstring;       // The language representation
    int birth_step;              // When this language first appeared
    std::string parent;          // Parent language bitstring
    int cluster_size_at_birth;   // Size of cluster when first recorded
    int max_cluster_size;        // Maximum cluster size observed
};

void run(int L, int B, double gamma, double alpha, double mu, int steps, int steps_to_record, int recording_skip, int min_cluster_size, const std::string &output_path)
{
    // Lattice initialization: all agents identical
    std::vector<std::vector<Agent>> lattice(L, std::vector<Agent>(L, Agent(B)));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice[i][j].language = std::vector<int>(B, 0);

    // Tracking language phylogeny - keep track of parent relationships
    std::map<std::string, std::string> language_parents; // language -> parent language
    
    // Record the initial language
    std::string initial_language = bitstring_to_string(lattice[0][0].language);
    language_parents[initial_language] = "";  // Root has no parent

    std::ofstream fout(output_path);
    // Write header
    fout << "step\tlanguage_id\tlargest_cluster_size\tparent_id\n";

    for (int step = 0; step < steps; ++step)
    {
        update(lattice, gamma, alpha, mu, L, B);

        if (step % 100 == 0)
            std::cout << "Step " << step << "/" << steps << "\r" << std::flush;

        // Record data at specified intervals and during the recording period
        if (step % recording_skip == 0 && step >= steps - steps_to_record) {
            auto cluster_data = get_language_cluster_sizes(lattice, L);
            
            // Process each language in the current timestep
            for (const auto& [language, cluster_sizes] : cluster_data) {
                // Find the largest cluster for this language
                int largest_cluster = 0;
                for (int size : cluster_sizes) {
                    largest_cluster = std::max(largest_cluster, size);
                }
                
                // Skip if the language doesn't meet the minimum cluster size
                if (largest_cluster < min_cluster_size) continue;
                
                // Check if this is a new language we haven't seen before
                if (language_parents.find(language) == language_parents.end()) {
                    // Find parent language by checking agents in the lattice
                    std::string parent_language = "";
                    bool found_parent = false;
                    
                    // Search for an agent with this language and check its parent
                    for (int i = 0; i < L && !found_parent; ++i) {
                        for (int j = 0; j < L && !found_parent; ++j) {
                            if (bitstring_to_string(lattice[i][j].language) == language && 
                                !lattice[i][j].parent_language.empty()) {
                                parent_language = lattice[i][j].parent_language;
                                found_parent = true;
                                break;
                            }
                        }
                    }
                    
                    // If we couldn't find a parent through direct tracking, use nearest known language
                    if (!found_parent || parent_language.empty()) {
                        // Convert language to vector<int> format for distance calculation
                        std::vector<int> lang_vec(B, 0);
                        for (size_t i = 0; i < language.size() && i < B; ++i) {
                            lang_vec[i] = (language[i] == '1') ? 1 : 0;
                        }
                        
                        // Find closest known language
                        int min_distance = B + 1;
                        
                        for (const auto& [known_lang, known_parent] : language_parents) {
                            // Convert known language to vector<int>
                            std::vector<int> known_vec(B, 0);
                            for (size_t i = 0; i < known_lang.size() && i < B; ++i) {
                                known_vec[i] = (known_lang[i] == '1') ? 1 : 0;
                            }
                            
                            // Calculate Hamming distance
                            int distance = 0;
                            for (int i = 0; i < B; ++i) {
                                if (lang_vec[i] != known_vec[i]) distance++;
                            }
                            
                            if (distance < min_distance) {
                                min_distance = distance;
                                parent_language = known_lang;
                            }
                        }
                    }
                    
                    // Record this new language
                    language_parents[language] = parent_language;
                }
                
                // Convert language and parent to integers for output
                std::vector<int> lang_vec(B, 0);
                for (size_t i = 0; i < language.size() && i < B; ++i) {
                    lang_vec[i] = (language[i] == '1') ? 1 : 0;
                }
                uint64_t lang_int = bitstring_to_int(lang_vec);
                
                uint64_t parent_int = 0;
                std::string parent_language = language_parents[language];
                if (!parent_language.empty()) {
                    std::vector<int> parent_vec(B, 0);
                    for (size_t i = 0; i < parent_language.size() && i < B; ++i) {
                        parent_vec[i] = (parent_language[i] == '1') ? 1 : 0;
                    }
                    parent_int = bitstring_to_int(parent_vec);
                }
                
                // Write current data: step, language_id, largest_cluster_size, parent_id
                fout << step << "\t" 
                     << lang_int << "\t"
                     << largest_cluster << "\t"
                     << (parent_language.empty() ? "" : std::to_string(parent_int)) << "\n";
            }
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
    int min_cluster_size = DEFAULT_MIN_CLUSTER_SIZE;
    
    if (argc > 1) L = std::stoi(argv[1]);
    if (argc > 2) B = std::stoi(argv[2]);
    if (argc > 3) steps = std::stoi(argv[3]);
    if (argc > 4) gamma = std::stod(argv[4]);
    if (argc > 5) alpha = std::stod(argv[5]);
    if (argc > 6) mu = std::stod(argv[6]);
    if (argc > 7) steps_to_record = std::stoi(argv[7]);
    if (argc > 8) recording_skip = std::stoi(argv[8]);
    if (argc > 9) min_cluster_size = std::stoi(argv[9]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream fname;
    fname << exeDir << "/outputs/phylogeny/L_" << L << "_g_" << gamma << "_a_" << alpha << "_B_" << B << "_mu_" << mu << "_minSize_" << min_cluster_size << ".tsv";

    // Create directory if it doesn't exist
    std::filesystem::path dir_path = std::filesystem::path(exeDir) / "outputs" / "phylogeny";
    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    run(L, B, gamma, alpha, mu, steps, steps_to_record, recording_skip, min_cluster_size, fname.str());

    return 0;
}