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
constexpr double DEFAULT_GAMMA = -0.05;
constexpr int DEFAULT_N = 1000;
constexpr int DEFAULT_L = 16;
constexpr int DEFAULT_N_ROUNDS = 500;
constexpr double DEFAULT_MU = 0.01;
constexpr int DEFAULT_CHILDREN_PER_SUCCESS = 2;
constexpr int DEFAULT_GENERATIONS = 1000;
constexpr int DEFAULT_MAX_DEPTH = 5;
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
    long long parent_id;  // ID of parent
    int birth_generation; // When this agent was born

    Agent(int L, long long parent = -1, int gen = 0)
        : language(L, 0), fitness(0.0), parent_id(parent), birth_generation(gen) {}
};

// Memory-efficient approach to track genetic similarity
class GeneticTracker
{
private:
    int current_gen;
    std::vector<long long> agent_to_parent; // Maps agent index to parent index
    std::vector<int> agent_to_gen;          // Maps agent index to birth generation
    int max_search_depth;                   // Maximum depth to search for ancestors

public:
    GeneticTracker(int N, int max_depth) : current_gen(0), max_search_depth(max_depth)
    {
        agent_to_parent.resize(N, -1);
        agent_to_gen.resize(N, 0);
    }

    void recordParentage(long long child_idx, long long parent_idx, int gen)
    {
        if (child_idx >= agent_to_parent.size())
        {
            size_t new_size = std::max(static_cast<size_t>(child_idx + 1), agent_to_parent.size() * 2);
            agent_to_parent.resize(new_size, -1);
            agent_to_gen.resize(new_size, 0);
        }
        agent_to_parent[child_idx] = parent_idx;
        agent_to_gen[child_idx] = gen;
        current_gen = gen;
    }

    // Calculate genetic distance between two agents
    int geneticDistance(long long idx1, long long idx2)
    {
        if (idx1 == idx2)
            return 0; // Same agent

        // Make sure indices are valid
        if (idx1 < 0 || idx2 < 0 ||
            idx1 >= agent_to_parent.size() ||
            idx2 >= agent_to_parent.size())
        {
            return -1; // Invalid - no relationship
        }

        // Keep track of ancestors for both agents
        std::vector<long long> ancestors1, ancestors2;
        long long curr1 = idx1, curr2 = idx2;
        int depth = 0;

        // Build ancestor lists (limited depth to save computation)
        while (curr1 >= 0 && depth < max_search_depth)
        {
            ancestors1.push_back(curr1);
            curr1 = agent_to_parent[curr1];
            depth++;
        }

        depth = 0;
        while (curr2 >= 0 && depth < max_search_depth)
        {
            ancestors2.push_back(curr2);
            curr2 = agent_to_parent[curr2];
            depth++;
        }

        // Check if one is ancestor of the other
        for (size_t i = 0; i < ancestors2.size(); i++)
        {
            if (ancestors2[i] == idx1)
            {
                return i + 1; // Direct line - distance is generations apart
            }
        }

        for (size_t i = 0; i < ancestors1.size(); i++)
        {
            if (ancestors1[i] == idx2)
            {
                return i + 1; // Direct line - distance is generations apart
            }
        }

        // Look for common ancestor
        for (size_t i = 0; i < ancestors1.size(); i++)
        {
            for (size_t j = 0; j < ancestors2.size(); j++)
            {
                if (ancestors1[i] == ancestors2[j] && ancestors1[i] >= 0)
                {
                    // Found common ancestor - total distance through this ancestor
                    return i + j + 2;
                }
            }
        }

        return -1; // No close relation found
    }

    void advanceGeneration()
    {
        current_gen++;
    }
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
void evolveLanguages(double gamma, int N, int L, int N_rounds, double mu,
                     int children_per_success, int generations, int max_depth,
                     const std::string &fitness_file, const std::string &languages_file)
{

    // Ensure N is divisible by children_per_success
    if (N % children_per_success != 0)
    {
        N = static_cast<int>(std::round(static_cast<double>(N) / children_per_success) * children_per_success);
        std::cout << "Adjusted N to " << N << " to be divisible by children_per_success\n";
    }

    // Initialize genetic tracker
    GeneticTracker genetic_tracker(N, max_depth);

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
            // Shuffle indices for random pairing
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
            std::shuffle(indices.begin(), indices.end(), gen);

            // Pair agents and compute fitness
            for (int i = 0; i < N; i += 2)
            {
                if (i + 1 >= N)
                    break; // Skip odd agent if N is odd

                int idx_a = indices[i];
                int idx_b = indices[i + 1];

                Agent &a = population[idx_a];
                Agent &b = population[idx_b];

                // Calculate genetic distance
                int genetic_dist = genetic_tracker.geneticDistance(idx_a, idx_b);

                // Calculate genetic similarity bonus (1 / distance)
                double genetic_bonus = 0.0;
                if (genetic_dist > 0)
                { // Only apply bonus if there's a genetic relationship
                    genetic_bonus = 1.0 / genetic_dist;
                }

                // Calculate fitness with only hamming distance and genetic bonus
                double fit_a = gamma * hamming(a.language, b.language) + genetic_bonus;
                double fit_b = gamma * hamming(b.language, a.language) + genetic_bonus;

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

        // Sort by fitness and select top N/children_per_success agents
        std::sort(population.begin(), population.end(),
                  [](const Agent &a, const Agent &b)
                  { return a.fitness > b.fitness; });

        int n_success = N / children_per_success;
        std::vector<Agent> next_gen;
        next_gen.reserve(N);

        // Create children and track parentage
        for (int i = 0; i < n_success; ++i)
        {
            for (int c = 0; c < children_per_success; ++c)
            {
                Agent child(L, i, generation + 1);
                child.language = mutate(population[i].language, mu);

                // Add child to next generation
                next_gen.push_back(std::move(child));

                // Record parentage (child's index will be i*children_per_success + c)
                long long child_idx = i * children_per_success + c;
                genetic_tracker.recordParentage(child_idx, i, generation + 1);
            }
        }

        // Replace population
        population = std::move(next_gen);

        // Advance tracker to next generation
        genetic_tracker.advanceGeneration();

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << (static_cast<double>(generation + 1) / (generations + 1) * 100.0) << "%\r" << std::flush;
    }

    // Close files
    fitness_out.close();
    langs_out.close();
}

int main(int argc, char *argv[])
{
    // Default parameters
    double gamma = DEFAULT_GAMMA;
    int N = DEFAULT_N;
    int L = DEFAULT_L;
    int N_rounds = DEFAULT_N_ROUNDS;
    double mu = DEFAULT_MU;
    int children_per_success = DEFAULT_CHILDREN_PER_SUCCESS;
    int generations = DEFAULT_GENERATIONS;
    int max_depth = DEFAULT_MAX_DEPTH;

    // Parse command line args
    if (argc > 1)
        gamma = std::stod(argv[1]);
    if (argc > 2)
        N = std::stoi(argv[2]);
    if (argc > 3)
        L = std::stoi(argv[3]);
    if (argc > 4)
        N_rounds = std::stoi(argv[4]);
    if (argc > 5)
        mu = std::stod(argv[5]);
    if (argc > 6)
        children_per_success = std::stoi(argv[6]);
    if (argc > 7)
        generations = std::stoi(argv[7]);
    if (argc > 8)
        max_depth = std::stoi(argv[8]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Generate output filenames with parameters
    std::ostringstream fitness_stream;
    fitness_stream << exeDir << "/outputs/fitness/g_" << gamma
                   << "_N_" << N << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string fitness_file = fitness_stream.str();

    std::ostringstream langs_stream;
    langs_stream << exeDir << "/outputs/languages/g_" << gamma
                 << "_N_" << N << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string languages_file = langs_stream.str();

    // Run evolution with simplified fitness
    evolveLanguages(gamma, N, L, N_rounds, mu, children_per_success, generations, max_depth,
                    fitness_file, languages_file);

    return 0;
}