import numpy as np
import matplotlib.pyplot as plt

def main():
    gamma = 10
    N = 1000
    L = 16
    mu = 0.01
    generation, max_fitness, avg_fitness = np.loadtxt(f"src/evolveHamming/outputs/fitness_g_{gamma}_N_{N}_L_{L}_mu_{mu}.tsv", delimiter='\t', unpack=True, skiprows=1)

    plt.plot(generation, max_fitness, label='Max Fitness')
    plt.plot(generation, avg_fitness, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'($\gamma$={gamma}, N={N}, L={L}, $\mu$={mu})')
    plt.legend()
    plt.grid()
    plt.savefig(f"src/evolveHamming/plots/fitness/timeseries_g_{gamma}_N_{N}_L_{L}_mu_{mu}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()