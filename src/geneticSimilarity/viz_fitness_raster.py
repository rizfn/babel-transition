import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from matplotlib.colors import LinearSegmentedColormap

def extract_params_from_filename(filename):
    """Extract gamma and max_depth from filename."""
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    max_depth_match = re.search(r'gdmax_([0-9]+)', filename)  # <-- updated here
    if gamma_match and max_depth_match:
        gamma = float(gamma_match.group(1))
        max_depth = int(max_depth_match.group(1))
        return gamma, max_depth
    else:
        return None, None

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
    fitness_files_pattern = "src/geneticSimilarity/outputs/betaNegative/fitness/g_*_N_*_L_*_mu_*.tsv"
    fitness_files = glob.glob(fitness_files_pattern)
    if not fitness_files:
        print(f"No files found matching pattern: {fitness_files_pattern}")
        return

    print(f"Found {len(fitness_files)} fitness files")
    results = []

    for filename in fitness_files:
        gamma, max_depth = extract_params_from_filename(filename)
        if gamma is not None and max_depth is not None:
            equilibrium_fitness = calculate_equilibrium_fitness(filename)
            results.append((gamma, max_depth, equilibrium_fitness))

    results = np.array(results)
    if results.size == 0:
        print("No valid results found.")
        return

    gammas = np.sort(np.unique(results[:, 0]))
    max_depths = np.sort(np.unique(results[:, 1]))

    fitness_grid = np.full((len(gammas), len(max_depths)), np.nan)
    for gamma, max_depth, fitness in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        max_depth_idx = np.where(max_depths == max_depth)[0][0]
        fitness_grid[gamma_idx, max_depth_idx] = fitness

    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        fitness_grid, cmap='viridis', aspect='auto',
        extent=[min(max_depths)-0.5, max(max_depths)+0.5, min(gammas)-0.5, max(gammas)+0.5],
        origin='lower'
    )

    # Annotate with actual fitness values
    for i, gamma in enumerate(gammas):
        for j, max_depth in enumerate(max_depths):
            if not np.isnan(fitness_grid[i, j]):
                value = fitness_grid[i, j]
                brightness = (value - np.nanmin(fitness_grid)) / (np.nanmax(fitness_grid) - np.nanmin(fitness_grid))
                text_color = 'white' if brightness > 0.5 else 'black'
                plt.text(max_depth, gamma, f"{value:.2f}", ha='center', va='center', color=text_color, fontweight='bold')

    cbar = plt.colorbar(im)
    cbar.set_label('Equilibrium Fitness (avg of last 100 generations)', rotation=270, labelpad=20)

    plt.xticks(max_depths)
    plt.yticks(gammas)
    ax = plt.gca()
    ax.set_xticks(max_depths, minor=True)
    ax.set_yticks(gammas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    plt.xlabel('max_depth')
    plt.ylabel('Gamma (hamming distance penalty)')
    plt.title('Equilibrium Fitness Across Gamma and max_depth')

    plt.savefig("src/geneticSimilarity/plots/fitness/betaNegative_heatmap.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()