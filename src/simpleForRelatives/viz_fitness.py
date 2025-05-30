import numpy as np
import matplotlib.pyplot as plt

def beta2params():
    gamma = 0
    alpha = 100
    max_depth = 10
    N = 1000
    L = 16
    mu = 0.01
    beta = 0.1
    generation, max_fitness, avg_fitness = np.loadtxt(f"src/simpleForRelatives/outputs/beta/fitness/g_{gamma}_a_{alpha}_gdmax_{max_depth}_N_{N}_b_{beta}_L_{L}_mu_{mu}.tsv", delimiter='\t', unpack=True, skiprows=1)

    plt.plot(generation, max_fitness, label='Max Fitness')
    plt.plot(generation, avg_fitness, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'($\gamma$={gamma}, $\\alpha$={alpha}, ${{g_d}}_{{max}}$={max_depth}, N={N}, L={L}, $\mu$={mu}, $\\beta$={beta})')
    plt.legend()
    plt.grid()
    plt.savefig(f"src/simpleForRelatives/plots/fitness/beta_{beta}_g_{gamma}_a_{alpha}_gdmax_{max_depth}_N_{N}_L_{L}_mu_{mu}.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    beta2params()
