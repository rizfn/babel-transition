#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <set>
#include <iomanip>
#include <numeric>
#include <map>
#include <unordered_map>
#include <functional> // For std::hash

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Default parameters
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 0.4;
constexpr int DEFAULT_N = 65536;
constexpr int DEFAULT_B = 16;
constexpr double DEFAULT_MU = 0.000001;
constexpr int DEFAULT_GENERATIONS = 1000;
constexpr int DEFAULT_LAST_GENS_TO_RECORD = 100;
constexpr int DEFAULT_RECORDING_SKIP = 10;

// Define a language as a bitstring
using Language = std::vector<int>;

// Define an agent
struct Agent {
    Language language;
    double fitness;
    int birth_generation;

    Agent(int B, int gen = 0)
        : language(B, 0), fitness(0.0), birth_generation(gen) {}
};

// Custom hash for std::vector<int>
struct VectorIntHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for (auto& i : v) {
            seed ^= std::hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Function to compute Hamming distance between two languages
int hamming(const Language& a, const Language& b) {
    int distance = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) distance++;
    }
    return distance;
}

// Function to count the number of shared 1s between two languages
int communicability(const Language& a, const Language& b) {
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 1 && b[i] == 1) {
            count++;
        }
    }
    return count;
}

// Calculate mean-field communicability (fraction of bits that can communicate with mean field)
double mean_field_communicability(const Language& language, const std::vector<double>& mean_field, int B) {
    double comm = 0.0;
    for (int i = 0; i < B; ++i) {
        if (language[i] == 1) {
            comm += mean_field[i];
        }
    }
    return comm / static_cast<double>(B);
}

// Calculate mean-field distance (equivalent to hamming distance with mean field)
double mean_field_distance(const Language& language, const std::vector<double>& mean_field, int B) {
    double distance = 0.0;
    for (int i = 0; i < B; ++i) {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(B);
}

// Function to mutate a language
Language mutate(const Language& lang, double mu, int B) {
    Language mutated = lang;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < B; ++i) {
        if (dis(gen) < mu) {
            mutated[i] = 1 - mutated[i]; // Flip the bit
        }
    }

    return mutated;
}

// Main evolution function
void evolveLanguages(double gamma, double alpha, int N, int B, double mu,
                     int generations,
                     int last_gens_to_record, int recording_skip,
                     const std::string& order_param_file)
{
    // Ensure N is even for pairing
    if (N % 2 != 0) {
        N += 1;
        std::cout << "Adjusted N to " << N << " to be even for pairing\n";
    }

    // Initialize population
    std::vector<Agent> population;
    population.reserve(N);
    for (int i = 0; i < N; ++i) {
        population.emplace_back(B, 0);
    }

    // Open file for order parameters
    std::ofstream order_out(order_param_file);
    order_out << "generation\tnum_languages\tlargest_cluster_size\n";

    // Evolution loop
    for (int generation = 0; generation <= generations; ++generation) {
        // 1. Calculate mean-field bitstring
        std::vector<double> mean_field(B, 0.0);
        for (const auto& agent : population) {
            for (int b = 0; b < B; ++b) {
                mean_field[b] += static_cast<double>(agent.language[b]);
            }
        }
        for (int b = 0; b < B; ++b) {
            mean_field[b] /= static_cast<double>(N);
        }

        // 2. Fitness evaluation using mean field
        double total_fitness = 0.0;
        for (auto& agent : population) {
            agent.fitness = 0.0;
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field, B);
            agent.fitness += global_fitness;
            double comm_fitness = alpha * mean_field_communicability(agent.language, mean_field, B);
            agent.fitness += comm_fitness;
            total_fitness += agent.fitness;
        }

        // Sort by fitness and select top 50%
        std::nth_element(population.begin(), population.begin() + N / 2, population.end(),
                         [](const Agent& a, const Agent& b) { return a.fitness > b.fitness; });
        std::sort(population.begin(), population.begin() + N / 2,
                  [](const Agent& a, const Agent& b) { return a.fitness > b.fitness; });

        int n_success = N / 2;
        std::vector<Agent> next_gen;
        next_gen.reserve(N);

        // Each successful agent produces two children
        for (int i = 0; i < n_success; ++i) {
            for (int c = 0; c < 2; ++c) {
                Agent child(B, generation + 1);
                child.language = mutate(population[i].language, mu, B);
                next_gen.push_back(std::move(child));
            }
        }

        // Replace population
        population = std::move(next_gen);

        // Record order parameters in the last gens
        if (generation > generations - last_gens_to_record && generation % recording_skip == 0) {
            // Count clusters (unique languages) and largest cluster size
            std::unordered_map<std::vector<int>, int, VectorIntHash> cluster_counts;
            cluster_counts.reserve(N);
            for (const auto& agent : population) {
                ++cluster_counts[agent.language];
            }
            int num_languages = static_cast<int>(cluster_counts.size());
            int largest_cluster_size = 0;
            for (const auto& kv : cluster_counts) {
                if (kv.second > largest_cluster_size) largest_cluster_size = kv.second;
            }
            order_out << generation << "\t" << num_languages << "\t" << largest_cluster_size << "\n";
        }

        if (generation % 100 == 0 || generation == generations) {
            std::cout << "Progress: " << std::fixed << std::setprecision(2)
                      << (static_cast<double>(generation + 1) / (generations + 1) * 100.0)
                      << "%\r" << std::flush;
        }
    }

    // Close file
    order_out.close();
}

int main(int argc, char* argv[]) {
    // gamma, alpha are the first two parameters
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    int N = DEFAULT_N;
    int B = DEFAULT_B;
    double mu = DEFAULT_MU;
    int generations = DEFAULT_GENERATIONS;

    int last_gens_to_record = DEFAULT_LAST_GENS_TO_RECORD;
    int recording_skip = DEFAULT_RECORDING_SKIP;
    int simNo = 0;

    // Parse command line args
    if (argc > 1) gamma = std::stod(argv[1]);
    if (argc > 2) alpha = std::stod(argv[2]);
    if (argc > 3) N = std::stoi(argv[3]);
    if (argc > 4) B = std::stoi(argv[4]);
    if (argc > 5) mu = std::stod(argv[5]);
    if (argc > 6) generations = std::stoi(argv[6]);
    if (argc > 7) last_gens_to_record = std::stoi(argv[7]);
    if (argc > 8) recording_skip = std::stoi(argv[8]);
    if (argc > 9) simNo = std::stoi(argv[9]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Output filename for order parameters, now with simNo
    std::ostringstream order_stream;
    order_stream << exeDir << "/outputs/rasterMu/orderParams/"
                 << "N_" << N << "_B_" << B
                 << "/g_" << gamma << "_a_" << alpha << "_mu_" << mu
                 << "_sim_" << simNo << ".tsv";
    std::string order_param_file = order_stream.str();

    std::cout << order_param_file << std::endl;

    // Run evolution
    evolveLanguages(gamma, alpha, N, B, mu, generations,
                    last_gens_to_record, recording_skip,
                    order_param_file);

    return 0;
}