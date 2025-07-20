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
#include <chrono>

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
constexpr int DEFAULT_N_STEPS = 10000;
constexpr double DEFAULT_GAMMA = 1;
constexpr double DEFAULT_ALPHA = 1.2;
constexpr double DEFAULT_MU = 0.001;
constexpr int DEFAULT_STEPS_TO_RECORD = 5000;
constexpr int DEFAULT_RECORDING_SKIP = 1000;
constexpr int DEFAULT_EQUILIBRIUM_STEPS = 10000;

struct Agent
{
    std::vector<int> language;
    double fitness = 0.0;
    bool immune = false;

    Agent(int B) : language(B, 0) {}
};

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

std::tuple<int, int> find_weakest_neighbor(
    const std::vector<std::vector<Agent>> &lattice, int cx, int cy, int L)
{
    double min_fitness = std::numeric_limits<double>::infinity();
    std::vector<std::pair<int, int>> weakest_neighbors;
    
    int ni[4] = {(cx + 1) % L, (cx - 1 + L) % L, cx, cx};
    int nj[4] = {cy, cy, (cy + 1) % L, (cy - 1 + L) % L};
    
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
    
    return {-1, -1};
}

void update(std::vector<std::vector<Agent>> &lattice, double gamma, double alpha, double mu, int L, int B)
{
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
    
    for (int b = 0; b < B; ++b)
    {
        mean_field[b] /= static_cast<double>(total_agents);
    }

    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            Agent &agent = lattice[i][j];
            agent.fitness = 0.0;
            
            double global_fitness = gamma * mean_field_distance(agent.language, mean_field, B);
            agent.fitness += global_fitness;
            
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

    std::vector<std::pair<int, int>> positions;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            positions.emplace_back(i, j);
    
    std::shuffle(positions.begin(), positions.end(), gen);

    for (auto &row : lattice)
        for (auto &agent : row)
            agent.immune = false;

    int trials = (L * L) / 2;
    for (int trial = 0; trial < trials && trial < positions.size(); ++trial)
    {
        auto [i, j] = positions[trial];
        
        if (lattice[i][j].immune)
            continue;
        
        auto [wi, wj] = find_weakest_neighbor(lattice, i, j, L);
        
        if (wi == -1 && wj == -1)
            continue;
        
        if (lattice[i][j].fitness > lattice[wi][wj].fitness)
        {
            lattice[wi][wj].language = lattice[i][j].language;
            lattice[i][j].immune = true;
            lattice[wi][wj].immune = true;
        }
    }

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


void run_hysteresis(int L, int B, double gamma, double alpha, double mu, int steps, 
                   int steps_to_record, int recording_skip, int equilibrium_steps, const std::string &output_dir)
{
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // Generate Unix timestamp
    std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    
    // Run 1: Start from all-ones initial condition
    std::vector<std::vector<Agent>> lattice1(L, std::vector<Agent>(L, Agent(B)));
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice1[i][j].language = std::vector<int>(B, 1);  // All ones
    
    std::string output1 = output_dir + "/hysteresis_all_ones_g_" + std::to_string(gamma) + 
                          "_a_" + std::to_string(alpha) + "_mu_" + std::to_string(mu) + 
                          "_" + timestamp + ".tsv";
    std::ofstream fout1(output1);
    
    for (int step = 0; step < steps; ++step)
    {
        update(lattice1, gamma, alpha, mu, L, B);
        
        if (step % 100 == 0)
        {
            double progress = ((double)(step + 1) / steps) * 100.0;
            std::cout << "All-ones progress: " << std::fixed << std::setprecision(2) << progress << "%\r" << std::flush;
        }
        
        if ((step >= steps - steps_to_record) && (step % recording_skip == 0))
        {
            fout1 << step << "\t";
            for (int i = 0; i < L; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    for (int b = 0; b < B; ++b)
                        fout1 << lattice1[i][j].language[b];
                    if (j < L - 1)
                        fout1 << ",";
                }
                if (i < L - 1)
                    fout1 << ";";
            }
            fout1 << "\n";
        }
    }
    fout1.close();
    std::cout << "All-ones progress: 100.00%\n" << std::flush;
    
    // Run 2: Start from all-zeros initial condition with equilibrium time
    std::vector<std::vector<Agent>> lattice2(L, std::vector<Agent>(L, Agent(B)));
    // Initialize with all zeros (default initialization)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            lattice2[i][j].language = std::vector<int>(B, 0);  // All zeros
    
    std::string output2 = output_dir + "/hysteresis_all_zeros_g_" + std::to_string(gamma) + 
                          "_a_" + std::to_string(alpha) + "_mu_" + std::to_string(mu) + 
                          "_" + timestamp + ".tsv";
    std::ofstream fout2(output2);
    
    int total_steps_all_zeros = equilibrium_steps + steps;
    
    for (int step = 0; step < total_steps_all_zeros; ++step)
    {
        update(lattice2, gamma, alpha, mu, L, B);
        
        if (step % 100 == 0)
        {
            double progress = ((double)(step + 1) / total_steps_all_zeros) * 100.0;
            std::cout << "All-zeros progress: " << std::fixed << std::setprecision(2) << progress << "%\r" << std::flush;
        }
        
        // Recording phase: only record during the last 'steps' iterations
        int recording_step = step - equilibrium_steps;
        if (recording_step >= 0 && 
            (recording_step >= steps - steps_to_record) && 
            (recording_step % recording_skip == 0))
        {
            fout2 << recording_step << "\t";
            for (int i = 0; i < L; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    for (int b = 0; b < B; ++b)
                        fout2 << lattice2[i][j].language[b];
                    if (j < L - 1)
                        fout2 << ",";
                }
                if (i < L - 1)
                    fout2 << ";";
            }
            fout2 << "\n";
        }
    }
    fout2.close();
    std::cout << "All-zeros progress: 100.00%\n" << std::flush;
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
    int equilibrium_steps = DEFAULT_EQUILIBRIUM_STEPS;
    
    if (argc > 1) L = std::stoi(argv[1]);
    if (argc > 2) B = std::stoi(argv[2]);
    if (argc > 3) steps = std::stoi(argv[3]);
    if (argc > 4) gamma = std::stod(argv[4]);
    if (argc > 5) alpha = std::stod(argv[5]);
    if (argc > 6) mu = std::stod(argv[6]);
    if (argc > 7) steps_to_record = std::stoi(argv[7]);
    if (argc > 8) recording_skip = std::stoi(argv[8]);
    if (argc > 9) equilibrium_steps = std::stoi(argv[9]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();
    std::string output_dir = exeDir + "/outputs/hysteresis/L_" + std::to_string(L) + "_B_" + std::to_string(B);

    run_hysteresis(L, B, gamma, alpha, mu, steps, steps_to_record, recording_skip, equilibrium_steps, output_dir);

    return 0;
}