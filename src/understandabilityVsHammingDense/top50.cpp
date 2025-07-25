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
constexpr int DEFAULT_N = 1000;
constexpr int DEFAULT_B = 16;
constexpr double DEFAULT_MU = 0.000001;
constexpr int DEFAULT_GENERATIONS = 1000;
constexpr int LANGUAGE_LAST_GENS_TO_RECORD = 200;
constexpr int LANGUAGE_RECORDING_SKIP = 1;

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
double mean_field_communicability(const Language& language, const std::vector<double>& mean_field) {
    double comm = 0.0;
    for (size_t i = 0; i < language.size(); ++i) {
        // Agent has bit i set AND mean field has probability > 0 for bit i
        if (language[i] == 1) {
            comm += mean_field[i];
        }
    }
    return comm / static_cast<double>(language.size());
}

// Calculate mean-field distance (equivalent to hamming distance with mean field)
double mean_field_distance(const Language& language, const std::vector<double>& mean_field) {
    double distance = 0.0;
    for (size_t i = 0; i < language.size(); ++i) {
        distance += std::abs(static_cast<double>(language[i]) - mean_field[i]);
    }
    return distance / static_cast<double>(language.size());
}

// Function to mutate a language
Language mutate(const Language& lang, double mu) {
    Language mutated = lang;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < lang.size(); ++i) {
        if (dis(gen) < mu) {
            mutated[i] = 1 - mutated[i]; // Flip the bit
        }
    }

    return mutated;
}

// Main evolution function
void evolveLanguages(double gamma, double alpha, int N, int B, double mu,
                     int generations,
                     const std::string& fitness_file, const std::string& languages_file)
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

    // Open files
    std::ofstream fitness_out(fitness_file);
    fitness_out << "generation\tmax_fitness\tavg_fitness\n";

    std::ofstream langs_out(languages_file);
    langs_out << "generation\tagent_id\tlanguage\n";

    // Evolution loop
    for (int generation = 0; generation <= generations; ++generation) {
        // 1. Calculate mean-field bitstring
        std::vector<double> mean_field(B, 0.0);
        for (const auto& agent : population) {
            for (int b = 0; b < B; ++b) {
                mean_field[b] += static_cast<double>(agent.language[b]);
            }
        }
        
        // Normalize to get probabilities
        for (int b = 0; b < B; ++b) {
            mean_field[b] /= static_cast<double>(N);
        }

        // 2. Fitness evaluation using mean field
        double total_fitness = 0.0;
        for (auto& agent : population) {
            agent.fitness = 0.0;
            
            // 2a. Global interaction with mean field (distance-based, want to be different)
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field);
            agent.fitness += global_fitness;
            
            // 2b. Global communicability with mean field (want to be communicable)
            double comm_fitness = alpha * mean_field_communicability(agent.language, mean_field);
            agent.fitness += comm_fitness;
            
            total_fitness += agent.fitness;
        }

        // Find max fitness and average for logging
        double max_fitness = -std::numeric_limits<double>::infinity();
        for (const auto& agent : population) {
            max_fitness = std::max(max_fitness, agent.fitness);
        }
        double avg_fitness = total_fitness / N;

        // Write to fitness file
        fitness_out << generation << "\t" << max_fitness << "\t" << avg_fitness << "\n";

        // Sort by fitness and select top 50%
        std::sort(population.begin(), population.end(),
                  [](const Agent& a, const Agent& b) { return a.fitness > b.fitness; });

        int n_success = N / 2;
        std::vector<Agent> next_gen;
        next_gen.reserve(N);

        // Each successful agent produces two children
        for (int i = 0; i < n_success; ++i) {
            for (int c = 0; c < 2; ++c) {
                Agent child(B, generation + 1);
                child.language = mutate(population[i].language, mu);
                next_gen.push_back(std::move(child));
            }
        }

        // Replace population
        population = std::move(next_gen);

        // Optionally record languages in the last gens
        if (generation > generations - LANGUAGE_LAST_GENS_TO_RECORD && generation % LANGUAGE_RECORDING_SKIP == 0) {
            // Sort languages lexicographically
            std::sort(population.begin(), population.end(),
                      [](const Agent& a, const Agent& b) {
                          for (size_t i = 0; i < a.language.size(); ++i) {
                              if (a.language[i] != b.language[i]) {
                                  return a.language[i] < b.language[i];
                              }
                          }
                          return false;
                      });

            for (size_t i = 0; i < population.size(); ++i) {
                langs_out << generation << "\t" << i << "\t";
                for (int bit : population[i].language) {
                    langs_out << bit;
                }
                langs_out << "\n";
            }
        }

        std::cout << "Progress: " << std::fixed << std::setprecision(2)
                  << (static_cast<double>(generation + 1) / (generations + 1) * 100.0)
                  << "%\r" << std::flush;
    }

    // Close files
    fitness_out.close();
    langs_out.close();
}

int main(int argc, char* argv[]) {
    // gamma, alpha are the first two parameters
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    int N = DEFAULT_N;
    int B = DEFAULT_B;
    double mu = DEFAULT_MU;
    int generations = DEFAULT_GENERATIONS;

    // Parse command line args
    if (argc > 1) gamma = std::stod(argv[1]);
    if (argc > 2) alpha = std::stod(argv[2]);
    if (argc > 3) N = std::stoi(argv[3]);
    if (argc > 4) B = std::stoi(argv[4]);
    if (argc > 5) mu = std::stod(argv[5]);
    if (argc > 6) generations = std::stoi(argv[6]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // // Output filenames
    // std::ostringstream fitness_stream;
    // fitness_stream << exeDir << "/outputs/top50/fitness/g_" << gamma
    //                << "_a_" << alpha << "_N_" << N
    //                << "_B_" << B << "_mu_" << mu << ".tsv";
    // std::string fitness_file = fitness_stream.str();

    // std::ostringstream langs_stream;
    // langs_stream << exeDir << "/outputs/top50/languages/g_" << gamma
    //              << "_a_" << alpha << "_N_" << N
    //              << "_B_" << B << "_mu_" << mu << ".tsv";
    // std::string languages_file = langs_stream.str();

        // Output filenames
    std::ostringstream fitness_stream;
    fitness_stream << exeDir << "/outputs/timeseries/fitness/g_" << gamma
                   << "_a_" << alpha << "_N_" << N
                   << "_B_" << B << "_mu_" << mu << ".tsv";
    std::string fitness_file = fitness_stream.str();

    std::ostringstream langs_stream;
    langs_stream << exeDir << "/outputs/timeseries/languages/g_" << gamma
                 << "_a_" << alpha << "_N_" << N
                 << "_B_" << B << "_mu_" << mu << ".tsv";
    std::string languages_file = langs_stream.str();

    // Run evolution
    evolveLanguages(gamma, alpha, N, B, mu, generations,
                    fitness_file, languages_file);

    return 0;
}