import numpy as np
import matplotlib.pyplot as plt
import os

def beta2params(gamma, alpha, N, beta, L, mu):
    fitness_file = f"src/simplicityVsHamming/outputs/beta/fitness/g_{gamma}_a_{alpha}_N_{N}_b_{beta}_L_{L}_mu_{mu}.tsv"
    if not os.path.exists(fitness_file):
        print(f"File not found: {fitness_file}")
        return

    generation, max_fitness, avg_fitness = np.loadtxt(fitness_file, delimiter='\t', unpack=True, skiprows=1)

    plt.plot(generation, max_fitness, label='Max Fitness')
    plt.plot(generation, avg_fitness, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'($\\gamma$={gamma}, $\\alpha$={alpha}, N={N}, L={L}, $\\mu$={mu}, $\\beta$={beta})')
    plt.legend()
    plt.grid()
    output_dir = "src/simplicityVsHamming/plots/fitness"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/beta_{beta}_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    beta2params(0, 1, 1000, 1, 16, 0.01)