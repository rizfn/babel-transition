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
constexpr double DEFAULT_GAMMA = -4.0;
constexpr int DEFAULT_N = 1000;
constexpr int DEFAULT_L = 16;
constexpr int DEFAULT_N_ROUNDS = 500;
constexpr double DEFAULT_MU = 0.01;
constexpr int DEFAULT_CHILDREN_PER_SUCCESS = 2;
constexpr int DEFAULT_GENERATIONS = 1000;

// Define a language as a bitstring
using Language = std::vector<int>;

// Define an agent
struct Agent {
    Language language;
    double fitness;
    std::vector<double> fitnesses;

    Agent(int L) : language(L, 0), fitness(0.0) {}
};

// Function to compute Hamming distance between two languages
int hamming(const Language& a, const Language& b) {
    int distance = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) distance++;
    }
    return distance;
}

// Function to count the number of 1s in a language
int sumBits(const Language& bits) {
    int sum = 0;
    for (int bit : bits) {
        sum += bit;
    }
    return sum;
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
void evolveLanguages(double gamma, int N, int L, int N_rounds, double mu, 
                    int children_per_success, int generations, 
                    const std::string& fitness_file, const std::string& languages_file) {
    
    // Ensure N is divisible by children_per_success
    if (N % children_per_success != 0) {
        N = static_cast<int>(std::round(static_cast<double>(N) / children_per_success) * children_per_success);
        std::cout << "Adjusted N to " << N << " to be divisible by children_per_success\n";
    }
    
    // Initialize population
    std::vector<Agent> population;
    population.reserve(N);
    for (int i = 0; i < N; ++i) {
        population.emplace_back(L);
    }
    
    // Open fitness file
    std::ofstream fitness_out(fitness_file);
    fitness_out << "generation\tmax_fitness\tavg_fitness\n";
    
    // Evolution loop
    for (int generation = 0; generation < generations; ++generation) {
        // Reset fitness for all agents
        for (auto& agent : population) {
            agent.fitness = 0;
            agent.fitnesses.clear();
        }
        
        // Run N_rounds of fitness evaluation
        for (int round = 0; round < N_rounds; ++round) {
            // Shuffle indices for random pairing
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
            std::shuffle(indices.begin(), indices.end(), gen);
            
            // Pair agents and compute fitness
            for (int i = 0; i < N; i += 2) {
                if (i + 1 >= N) break; // Skip odd agent if N is odd
                
                int idx_a = indices[i];
                int idx_b = indices[i + 1];
                
                Agent& a = population[idx_a];
                Agent& b = population[idx_b];
                
                double fit_a = sumBits(a.language) - gamma * hamming(a.language, b.language);
                double fit_b = sumBits(b.language) - gamma * hamming(b.language, a.language);
                
                a.fitnesses.push_back(fit_a);
                b.fitnesses.push_back(fit_b);
            }
        }
        
        // Calculate average fitness for each agent
        double total_fitness = 0.0;
        for (auto& agent : population) {
            if (!agent.fitnesses.empty()) {
                double sum = 0;
                for (double f : agent.fitnesses) {
                    sum += f;
                }
                agent.fitness = sum / agent.fitnesses.size();
                total_fitness += agent.fitness;
            }
        }
        
        // Find max fitness and average for logging
        double max_fitness = -std::numeric_limits<double>::infinity();
        for (const auto& agent : population) {
            max_fitness = std::max(max_fitness, agent.fitness);
        }
        double avg_fitness = total_fitness / N;
        
        // Write to fitness file
        fitness_out << generation << "\t" << max_fitness << "\t" << avg_fitness << "\n";
        
        // Sort by fitness and select top N/children_per_success agents
        std::sort(population.begin(), population.end(), 
                 [](const Agent& a, const Agent& b) { return a.fitness > b.fitness; });
        
        int n_success = N / children_per_success;
        std::vector<Agent> next_gen;
        next_gen.reserve(N);
        
        // Create children
        for (int i = 0; i < n_success; ++i) {
            for (int c = 0; c < children_per_success; ++c) {
                Agent child(L);
                child.language = mutate(population[i].language, mu);
                next_gen.push_back(std::move(child));
            }
        }
        
        // Replace population
        population = std::move(next_gen);

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << (
            static_cast<double>(generation + 1) / generations * 100.0) << "%\r" << std::flush;
    }
    
    // Close fitness file
    fitness_out.close();
    
    std::ofstream langs_out(languages_file);
    langs_out << "agent_id\tlanguage\n";
    
    // Sort languages lexicographically
    std::sort(population.begin(), population.end(), 
             [](const Agent& a, const Agent& b) {
                 for (size_t i = 0; i < a.language.size(); ++i) {
                     if (a.language[i] != b.language[i]) {
                         return a.language[i] < b.language[i];
                     }
                 }
                 return false; // Equal languages
             });
    
    for (size_t i = 0; i < population.size(); ++i) {
        langs_out << i << "\t";
        for (int bit : population[i].language) {
            langs_out << bit;
        }
        langs_out << "\n";
    }    
    langs_out.close();
}

int main(int argc, char* argv[]) {
    // Default parameters
    double gamma = DEFAULT_GAMMA;
    int N = DEFAULT_N;
    int L = DEFAULT_L;
    int N_rounds = DEFAULT_N_ROUNDS;
    double mu = DEFAULT_MU;
    int children_per_success = DEFAULT_CHILDREN_PER_SUCCESS;
    int generations = DEFAULT_GENERATIONS;
    
    // Parse command line args
    if (argc > 1) gamma = std::stod(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) L = std::stoi(argv[3]);
    if (argc > 4) N_rounds = std::stoi(argv[4]);
    if (argc > 5) mu = std::stod(argv[5]);
    if (argc > 6) children_per_success = std::stoi(argv[6]);
    if (argc > 7) generations = std::stoi(argv[7]);
    
    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    
    // Generate output filenames with parameters
    std::ostringstream fitness_stream;
    fitness_stream << exeDir << "/outputs/fitness_g_" << gamma << "_N_" << N 
                  << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string fitness_file = fitness_stream.str();
    
    std::ostringstream langs_stream;
    langs_stream << exeDir << "/outputs/languages_g_" << gamma << "_N_" << N 
                << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string languages_file = langs_stream.str();
        
    // Run evolution
    evolveLanguages(gamma, N, L, N_rounds, mu, children_per_success, generations, 
                   fitness_file, languages_file);
    
    return 0;
}