import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from matplotlib.colors import LinearSegmentedColormap

def extract_params_from_filename(filename):
    """Extract parameter values from filename."""
    # Extract gamma and alpha using regex
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha_match = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    
    if gamma_match and alpha_match:
        gamma = float(gamma_match.group(1))
        alpha = float(alpha_match.group(1))
        return gamma, alpha
    else:
        return None, None

def calculate_equilibrium_fitness(filename):
    """Calculate equilibrium fitness from last 100 generations."""
    try:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        # Get the average fitness column (index 2)
        avg_fitness = data[:, 2]
        
        # Calculate average of last 100 generations
        if len(avg_fitness) >= 100:
            equilibrium_fitness = np.mean(avg_fitness[-100:])
        else:
            equilibrium_fitness = np.mean(avg_fitness)  # Use all data if less than 100 generations
            
        return equilibrium_fitness
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return np.nan

def main():
    # Path to fitness files
    fitness_files_pattern = "src/geneticSimilaritySimplicity/outputs/fitness/g_*_a_*_N_*_L_*_mu_*.tsv"
    
    # Get all fitness files
    fitness_files = glob.glob(fitness_files_pattern)
    
    if not fitness_files:
        print(f"No files found matching pattern: {fitness_files_pattern}")
        return
    
    print(f"Found {len(fitness_files)} fitness files")
    
    # Extract parameters and calculate equilibrium fitness for each file
    results = []
    
    for filename in fitness_files:
        gamma, alpha = extract_params_from_filename(filename)
        if gamma is not None and alpha is not None:
            equilibrium_fitness = calculate_equilibrium_fitness(filename)
            results.append((gamma, alpha, equilibrium_fitness))
    
    # Convert to numpy arrays for easier manipulation
    results = np.array(results)
    
    # Get unique values of gamma and alpha (sorted)
    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))
    
    # Create a 2D grid for the heatmap
    fitness_grid = np.zeros((len(gammas), len(alphas)))
    fitness_grid.fill(np.nan)  # Fill with NaN initially
    
    # Map each result to the correct position in the grid
    for gamma, alpha, fitness in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        fitness_grid[gamma_idx, alpha_idx] = fitness
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap with gammas on y-axis and alphas on x-axis
    im = plt.imshow(fitness_grid, cmap='viridis', aspect='auto',
                   extent=[min(alphas)-0.5, max(alphas)+0.5, min(gammas)-0.5, max(gammas)+0.5],
                   origin='lower')
    
    # Annotate with actual fitness values
    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            if not np.isnan(fitness_grid[i, j]):
                # Choose text color based on fitness value for readability
                value = fitness_grid[i, j]
                brightness = (value - np.nanmin(fitness_grid)) / (np.nanmax(fitness_grid) - np.nanmin(fitness_grid))
                text_color = 'white' if brightness > 0.5 else 'black'
                
                plt.text(alpha, gamma, f"{value:.2f}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    
    # Create colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Equilibrium Fitness (avg of last 100 generations)', rotation=270, labelpad=20)
    
    # Set custom tick positions to match actual parameter values
    plt.xticks(alphas)
    plt.yticks(gammas)
    
    # Add grid lines at tick positions
    ax = plt.gca()
    ax.set_xticks(alphas, minor=True)
    ax.set_yticks(gammas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Set labels and title
    plt.xlabel('Alpha (relatedness bonus)')
    plt.ylabel('Gamma (hamming distance penalty)')
    plt.title('Equilibrium Fitness Across Parameter Space')
        
    # Save the plot
    plt.savefig("src/geneticSimilaritySimplicity/plots/fitness/equilibrium_fitness_heatmap.png", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()