import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from matplotlib.colors import LinearSegmentedColormap

def extract_params_from_filename(filename):
    """Extract gamma, alpha, max_depth, N, beta, L, mu from filename."""
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha_match = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    max_depth_match = re.search(r'gdmax_([0-9]+)', filename)
    N_match = re.search(r'N_([0-9]+)', filename)
    beta_match = re.search(r'b_([+-]?\d+\.?\d*)_', filename)
    L_match = re.search(r'L_([0-9]+)', filename)
    mu_match = re.search(r'mu_([0-9.]+)(?:_|\.tsv)', filename)
    if gamma_match and alpha_match and max_depth_match and N_match and beta_match and L_match and mu_match:
        gamma = float(gamma_match.group(1))
        alpha = float(alpha_match.group(1))
        max_depth = int(max_depth_match.group(1))
        N = int(N_match.group(1))
        beta = float(beta_match.group(1))
        L = int(L_match.group(1))
        mu = float(mu_match.group(1))
        return gamma, alpha, max_depth, N, beta, L, mu
    else:
        return None, None, None, None, None, None, None

def calculate_equilibrium_fitness(filename):
    """Calculate equilibrium fitness from last 100 generations."""
    try:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        avg_fitness = data[:, 2]
        if len(avg_fitness) >= 100:
            equilibrium_fitness = np.mean(avg_fitness[-100:])
        else:
            equilibrium_fitness = np.mean(avg_fitness)
        return equilibrium_fitness
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return np.nan

def main():
    # Path to fitness files (adjust path as needed)
    fitness_files_pattern = "src/simpleForRelatives/outputs/beta/fitness/g_*_a_*_gdmax_*_N_*_b_*_L_*_mu_*.tsv"
    fitness_files = glob.glob(fitness_files_pattern)
    if not fitness_files:
        print(f"No files found matching pattern: {fitness_files_pattern}")
        return

    print(f"Found {len(fitness_files)} fitness files")
    results = []

    for filename in fitness_files:
        gamma, alpha, max_depth, N, beta, L, mu = extract_params_from_filename(filename)
        if gamma is not None and alpha is not None and max_depth is not None:
            equilibrium_fitness = calculate_equilibrium_fitness(filename)
            results.append((gamma, alpha, max_depth, equilibrium_fitness, N, beta, L, mu))

    results = np.array(results)
    if results.size == 0:
        print("No valid results found.")
        return

    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))
    max_depths = np.unique(results[:, 2])
    N = int(results[0, 4])
    beta = results[0, 5]
    L = int(results[0, 6])
    mu = results[0, 7]
    max_depth = int(results[0, 2])

    fitness_grid = np.full((len(alphas), len(gammas)), np.nan)
    for gamma, alpha, _, fitness, *_ in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        fitness_grid[alpha_idx, gamma_idx] = fitness

    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        fitness_grid, cmap='viridis', aspect='auto',
        extent=[min(gammas)-0.5, max(gammas)+0.5, min(alphas)-0.5, max(alphas)+0.5],
        origin='lower'
    )

    # Annotate with actual fitness values
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(fitness_grid[i, j]):
                value = fitness_grid[i, j]
                vmin, vmax = np.nanmin(fitness_grid), np.nanmax(fitness_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                plt.text(gamma, alpha, f"{value:.2f}", ha='center', va='center', color=text_color, fontweight='bold')

    cbar = plt.colorbar(im)
    cbar.set_label('Equilibrium Fitness (avg of last 100 generations)', rotation=270, labelpad=20)

    plt.xticks(gammas)
    plt.yticks(alphas)
    ax = plt.gca()
    ax.set_xticks(gammas, minor=True)
    ax.set_yticks(alphas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    plt.xlabel('Gamma (hamming distance penalty)')
    plt.ylabel('Alpha (simplicity bonus)')
    plt.title('Equilibrium Fitness Across Alpha and Gamma')

    output_dir = "src/simpleForRelatives/plots/fitness"
    os.makedirs(output_dir, exist_ok=True)
    fname = (f"{output_dir}/beta_{beta}_heatmap_N_{N}_L_{L}_gdmax_{max_depth}_mu_{mu}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()