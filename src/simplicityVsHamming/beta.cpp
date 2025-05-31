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

static auto _ = []()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Default parameters
constexpr double DEFAULT_GAMMA = 0;
constexpr double DEFAULT_ALPHA = 1;
constexpr int DEFAULT_N = 1000;
constexpr int DEFAULT_L = 16;
constexpr int DEFAULT_N_ROUNDS = 500;
constexpr double DEFAULT_MU = 0.01;
constexpr double DEFAULT_BETA = 1;
constexpr int DEFAULT_GENERATIONS = 1000;
constexpr int LANGUAGE_LAST_GENS_TO_RECORD = 100;
constexpr int LANGUAGE_RECORDING_SKIP = 10;

// Define a language as a bitstring
using Language = std::vector<int>;

// Define an agent
struct Agent
{
    Language language;
    double fitness;
    std::vector<double> fitnesses;
    long long parent_id;      // ID of parent
    int birth_generation;     // When this agent was born

    Agent(int L, long long parent = -1, int gen = 0)
        : language(L, 0), fitness(0.0), parent_id(parent), birth_generation(gen)
    {}
};

// Function to compute Hamming distance between two languages
int hamming(const Language &a, const Language &b)
{
    int distance = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
            distance++;
    }
    return distance;
}

// Function to count the number of 1s in a language
int sumBits(const Language &bits)
{
    int sum = 0;
    for (int bit : bits)
    {
        sum += bit;
    }
    return sum;
}

// Function to mutate a language
Language mutate(const Language &lang, double mu)
{
    Language mutated = lang;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < lang.size(); ++i)
    {
        if (dis(gen) < mu)
        {
            mutated[i] = 1 - mutated[i]; // Flip the bit
        }
    }

    return mutated;
}

// Main evolution function
void evolveLanguages(double gamma, double alpha, int N, int L, int N_rounds, double mu,
                     double beta, int generations,
                     const std::string &fitness_file, const std::string &languages_file)
{
    // Initialize population
    std::vector<Agent> population;
    population.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        population.emplace_back(L, -1, 0);
    }

    // Open files
    std::ofstream fitness_out(fitness_file);
    fitness_out << "generation\tmax_fitness\tavg_fitness\n";

    std::ofstream langs_out(languages_file);
    langs_out << "generation\tagent_id\tlanguage\n";

    // Evolution loop
    for (int generation = 0; generation <= generations; ++generation)
    {
        // Reset fitness for all agents
        for (auto &agent : population)
        {
            agent.fitness = 0;
            agent.fitnesses.clear();
        }

        // Run N_rounds of fitness evaluation
        for (int round = 0; round < N_rounds; ++round)
        {
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);

            for (int i = 0; i < N; i += 2)
            {
                if (i + 1 >= N)
                    break;

                int idx_a = indices[i];
                int idx_b = indices[i + 1];

                Agent &a = population[idx_a];
                Agent &b = population[idx_b];

                int d_hamming = hamming(a.language, b.language);

                double fit_a = alpha * (static_cast<double>(sumBits(a.language)) / L)
                             + gamma * (static_cast<double>(d_hamming) / L);
                double fit_b = alpha * (static_cast<double>(sumBits(b.language)) / L)
                             + gamma * (static_cast<double>(d_hamming) / L);

                a.fitnesses.push_back(fit_a);
                b.fitnesses.push_back(fit_b);
            }
        }

        // Calculate average fitness for each agent
        double total_fitness = 0.0;
        for (auto &agent : population)
        {
            if (!agent.fitnesses.empty())
            {
                double sum = 0;
                for (double f : agent.fitnesses)
                {
                    sum += f;
                }
                agent.fitness = sum / agent.fitnesses.size();
                total_fitness += agent.fitness;
            }
        }

        // Find max fitness and average for logging
        double max_fitness = -std::numeric_limits<double>::infinity();
        for (const auto &agent : population)
        {
            max_fitness = std::max(max_fitness, agent.fitness);
        }
        double avg_fitness = total_fitness / N;

        // Write to fitness file
        fitness_out << generation << "\t" << max_fitness << "\t" << avg_fitness << "\n";

        if (generation > generations - LANGUAGE_LAST_GENS_TO_RECORD && generation % LANGUAGE_RECORDING_SKIP == 0)
        {
            // Sort languages lexicographically
            std::sort(population.begin(), population.end(),
                      [](const Agent &a, const Agent &b)
                      {
                          for (size_t i = 0; i < a.language.size(); ++i)
                          {
                              if (a.language[i] != b.language[i])
                              {
                                  return a.language[i] < b.language[i];
                              }
                          }
                          return false; // Equal languages
                      });

            for (size_t i = 0; i < population.size(); ++i)
            {
                langs_out << generation << "\t" << i << "\t";
                for (int bit : population[i].language)
                {
                    langs_out << bit;
                }
                langs_out << "\n";
            }
        }

        // Create the next generation using fitness-weighted selection
        std::vector<Agent> next_gen;
        next_gen.reserve(N);

        // Calculate selection weights based on fitness
        std::vector<double> weights(N);
        for (int i = 0; i < N; ++i)
        {
            weights[i] = exp(beta * population[i].fitness);
        }

        // Create discrete distribution based on weights
        std::discrete_distribution<int> selection(weights.begin(), weights.end());

        // Generate N children
        for (int i = 0; i < N; ++i)
        {
            // Select parent according to fitness-weighted distribution
            int parent_idx = selection(gen);

            // Create child
            Agent child(L, parent_idx, generation + 1);
            child.language = mutate(population[parent_idx].language, mu);

            // Add child to next generation
            next_gen.push_back(std::move(child));
        }

        // Replace population
        population = std::move(next_gen);

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << (static_cast<double>(generation + 1) / (generations + 1) * 100.0) << "%\r" << std::flush;
    }

    // Close files
    fitness_out.close();
    langs_out.close();
}


int main(int argc, char *argv[])
{
    // gamma, alpha are the first two parameters
    double gamma = DEFAULT_GAMMA;
    double alpha = DEFAULT_ALPHA;
    int N = DEFAULT_N;
    int L = DEFAULT_L;
    int N_rounds = DEFAULT_N_ROUNDS;
    double mu = DEFAULT_MU;
    double beta = DEFAULT_BETA;
    int generations = DEFAULT_GENERATIONS;

    // Parse command line args
    if (argc > 1)
        gamma = std::stod(argv[1]);
    if (argc > 2)
        alpha = std::stod(argv[2]);
    if (argc > 3)
        N = std::stoi(argv[3]);
    if (argc > 4)
        L = std::stoi(argv[4]);
    if (argc > 5)
        N_rounds = std::stoi(argv[5]);
    if (argc > 6)
        mu = std::stod(argv[6]);
    if (argc > 7)
        beta = std::stod(argv[7]);
    if (argc > 8)
        generations = std::stoi(argv[8]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Output filenames (no max_depth)
    std::ostringstream fitness_stream;
    fitness_stream << exeDir << "/outputs/beta/fitness/g_" << gamma
                   << "_a_" << alpha << "_N_" << N 
                   << "_b_" << beta << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string fitness_file = fitness_stream.str();

    std::ostringstream langs_stream;
    langs_stream << exeDir << "/outputs/beta/languages/g_" << gamma
                 << "_a_" << alpha << "_N_" << N 
                 << "_b_" << beta << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string languages_file = langs_stream.str();

    // Run evolution
    evolveLanguages(gamma, alpha, N, L, N_rounds, mu, beta, generations,
                    fitness_file, languages_file);

    return 0;
}