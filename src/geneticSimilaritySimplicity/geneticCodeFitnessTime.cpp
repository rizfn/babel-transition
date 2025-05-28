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
#include <array>

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
constexpr double DEFAULT_GAMMA = -4.0;
constexpr double DEFAULT_ALPHA = 1.0;
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

// Define a genetic code as an array of segment IDs
using GeneticCode = std::array<long long, DEFAULT_MAX_DEPTH>;

// Define an agent
struct Agent
{
    Language language;
    double fitness;
    std::vector<double> fitnesses;
    long long parent_id;      // ID of parent
    int birth_generation;     // When this agent was born
    GeneticCode genetic_code; // Genetic code for efficient relationship tracking

    Agent(int L, long long parent = -1, int gen = 0)
        : language(L, 0), fitness(0.0), parent_id(parent), birth_generation(gen)
    {
        // Initialize genetic code to all -1 (no ancestry)
        genetic_code.fill(-1);
    }
};

// Memory-efficient approach to track genetic similarity using coded segments
class GeneticTracker
{
private:
    int current_gen;
    long long next_segment_id; // Unique ID for new lineage segments

public:
    GeneticTracker() : current_gen(0), next_segment_id(0) {}

    // Generate a new genetic code for a child based on parent's code
    GeneticCode generateChildCode(const GeneticCode& parent_code)
    {
        GeneticCode child_code;
        
        // First position always gets a unique ID for this child
        child_code[0] = next_segment_id++;
        
        // Copy parent's code, shifted one position
        for (int i = 1; i < DEFAULT_MAX_DEPTH; i++)
        {
            child_code[i] = parent_code[i - 1];
        }
        
        return child_code;
    }

    // Calculate genetic distance between two agents based on their genetic codes
    int geneticDistance(const GeneticCode& code1, const GeneticCode& code2)
    {
        // If agents have the same first segment, they are the same agent
        if (code1[0] == code2[0] && code1[0] != -1)
            return 0;

        // If any segment matches, find the shallowest match
        for (int i = 0; i < DEFAULT_MAX_DEPTH; i++)
        {
            long long segment1 = code1[i];
            if (segment1 == -1) continue; // Skip uninitialized segments
            
            for (int j = 0; j < DEFAULT_MAX_DEPTH; j++)
            {
                long long segment2 = code2[j];
                if (segment2 == -1) continue; // Skip uninitialized segments
                
                if (segment1 == segment2)
                {
                    // Found a common ancestor
                    // The distance is the sum of depths plus 2
                    // (i steps from agent1 to ancestor, j steps from agent2 to ancestor)
                    return i + j + 2;
                }
            }
        }
        
        return -1; // No relationship found within tracked depth
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
    GeneticTracker genetic_tracker;

    // Initialize population
    std::vector<Agent> population;
    population.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        population.emplace_back(L, -1, 0);
        // First generation gets unique genetic codes in first position only
        population[i].genetic_code[0] = i; // Use agent index as first segment ID
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

                // Calculate genetic distance using optimized approach
                int genetic_dist = genetic_tracker.geneticDistance(a.genetic_code, b.genetic_code);

                // Calculate genetic similarity bonus (alpha / distance)
                double genetic_bonus = 0.0;
                if (genetic_dist > 0)
                { // Only apply bonus if there's a genetic relationship
                    genetic_bonus = alpha / genetic_dist;
                }

                // Calculate fitness with genetic bonus
                double fit_a = sumBits(a.language) - gamma * hamming(a.language, b.language) + genetic_bonus;
                double fit_b = sumBits(b.language) - gamma * hamming(b.language, a.language) + genetic_bonus;

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
                
                // Generate child's genetic code based on parent's code
                child.genetic_code = genetic_tracker.generateChildCode(population[i].genetic_code);

                // Add child to next generation
                next_gen.push_back(std::move(child));

                // We still record parent ID for compatibility
                long long child_idx = i * children_per_success + c;
                child.parent_id = i;
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
    double alpha = DEFAULT_ALPHA; // Default alpha value for genetic similarity bonus
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
        alpha = std::stod(argv[2]); // Parameter for genetic similarity
    if (argc > 3)
        N = std::stoi(argv[3]);
    if (argc > 4)
        L = std::stoi(argv[4]);
    if (argc > 5)
        N_rounds = std::stoi(argv[5]);
    if (argc > 6)
        mu = std::stod(argv[6]);
    if (argc > 7)
        children_per_success = std::stoi(argv[7]);
    if (argc > 8)
        generations = std::stoi(argv[8]);
    if (argc > 9)
        max_depth = std::stoi(argv[9]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    // Generate output filenames with parameters
    std::ostringstream fitness_stream;
    fitness_stream << exeDir << "/outputs/fitness/g_" << gamma << "_a_" << alpha
                   << "_N_" << N << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string fitness_file = fitness_stream.str();

    std::ostringstream langs_stream;
    langs_stream << exeDir << "/outputs/languages/g_" << gamma << "_a_" << alpha
                 << "_N_" << N << "_L_" << L << "_mu_" << mu << ".tsv";
    std::string languages_file = langs_stream.str();

    // Run evolution with optimized genetic tracking
    evolveLanguages(gamma, alpha, N, L, N_rounds, mu, children_per_success, generations, max_depth,
                    fitness_file, languages_file);

    return 0;
}