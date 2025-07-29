import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, N, B, mu from top50-style filename."""
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha_match = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    N_match = re.search(r'N_([0-9]+)', filename)
    B_match = re.search(r'B_([0-9]+)', filename)
    mu_match = re.search(r'mu_([0-9.]+)(?:_|\.tsv)', filename)
    
    if gamma_match and alpha_match and N_match and B_match and mu_match:
        gamma = float(gamma_match.group(1))
        alpha = float(alpha_match.group(1))
        N = int(N_match.group(1))
        B = int(B_match.group(1))
        mu = float(mu_match.group(1))
        return gamma, alpha, N, B, mu
    else:
        return None, None, None, None, None

def load_languages(filename):
    """Load languages from a TSV file and return as binary arrays from all time steps."""
    try:
        df = pd.read_csv(filename, sep='\t', dtype={'language': str})
        
        # Use all generations instead of just the last one
        languages = []
        for lang_str in df['language']:
            lang_str = str(lang_str)
            lang_array = np.array([int(bit) for bit in lang_str])
            languages.append(lang_array)
        return np.array(languages)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return np.array([])

def count_ones_in_population(languages):
    """Count the average number of 1s across all bitstrings in the population."""
    if len(languages) == 0:
        return 0
    
    total_ones = 0
    total_bits = 0
    
    for lang in languages:
        ones_count = np.sum(lang)
        total_ones += ones_count
        total_bits += len(lang)
    
    return total_ones / total_bits if total_bits > 0 else 0

def get_ones_distribution(languages):
    """Get the distribution of 1s counts in all bitstrings in the population."""
    if len(languages) == 0:
        return np.array([])
    
    ones_counts = []
    for lang in languages:
        ones_count = np.sum(lang)
        ones_counts.append(ones_count)
    
    return np.array(ones_counts)

def load_meanfield_data(N, B, gamma):
    """Load all populations and compute both average 1s counts and distributions for each alpha-mu combination."""
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/top50/languages/g_{gamma}_a_*_N_{N}_B_{B}_mu_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}, {}
    
    print(f"Found {len(files)} files matching N={N}, B={B}, gamma={gamma}")
    
    ones_data = {}
    distribution_data = {}
    
    for filename in tqdm(files, desc="Processing language data"):
        gamma_file, alpha, N_file, B_file, mu = extract_params_from_filename(filename)
        
        if gamma_file is None or alpha is None or mu is None:
            continue
            
        # Double-check that the extracted parameters match what we're looking for
        if N_file != N or B_file != B or gamma_file != gamma:
            continue
            
        try:
            languages = load_languages(filename)
            if len(languages) > 0:
                # Compute average 1s count
                avg_ones = count_ones_in_population(languages)
                ones_data[(alpha, mu)] = {
                    'avg_ones': avg_ones,
                    'alpha': alpha,
                    'mu': mu
                }
                
                # Compute 1s distribution
                ones_counts = get_ones_distribution(languages)
                distribution_data[(alpha, mu)] = {
                    'ones_counts': ones_counts,
                    'alpha': alpha,
                    'mu': mu
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return ones_data, distribution_data

def create_ones_heatmap(ones_data, N, B, gamma):
    """Create a heatmap of average number of 1s organized by alpha and mu."""
    if not ones_data:
        print("No data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in ones_data.keys()))
    mus = sorted(set(key[1] for key in ones_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create 2D array for heatmap
    heatmap_data = np.full((len(alphas), len(mus)), np.nan)
    
    for i, alpha in enumerate(alphas):
        for j, mu in enumerate(mus):
            if (alpha, mu) in ones_data:
                heatmap_data[i, j] = ones_data[(alpha, mu)]['avg_ones']
    
    # Create the heatmap using pcolormesh for better log scale support
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    MU_MESH, ALPHA_MESH = np.meshgrid(mus, alphas)
    
    im = ax.pcolormesh(MU_MESH, ALPHA_MESH, heatmap_data, cmap='viridis', 
                       vmin=0, vmax=1, shading='auto')
    ax.set_xscale('log')
    
    # Add text annotations with the values
    for i, alpha in enumerate(alphas):
        for j, mu in enumerate(mus):
            if not np.isnan(heatmap_data[i, j]):
                text = f'{heatmap_data[i, j]:.3f}'
                ax.text(mu, alpha, text, ha="center", va="center", 
                       color="white" if heatmap_data[i, j] < 0.5 else "black",
                       fontsize=8, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(mus)
    ax.set_xticklabels([f"{mu:.1e}" if mu < 0.001 else f"{mu:.3f}" for mu in mus], rotation=45)
    ax.set_yticks(alphas)
    ax.set_yticklabels([f'{alpha:.2f}' for alpha in alphas])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Number of 1s per Bitstring', rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('Mu (Mutation Rate)', fontsize=12)
    ax.set_ylabel('Alpha (Local Interaction Strength)', fontsize=12)
    ax.set_title(f'Average Number of 1s per Bitstring (All Time Steps)\n(N={N}, B={B}, γ={gamma})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "src/understandabilityVsHammingDense/plots/ones_fraction"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/ones_heatmap_alltimesteps_N_{N}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Heatmap saved to: {fname}")

def create_distribution_grid(distribution_data, N, B, gamma):
    """Create a grid plot of 1s distributions organized by mu (x-axis) and alpha (y-axis)."""
    if not distribution_data:
        print("No distribution data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in distribution_data.keys()))
    mus = sorted(set(key[1] for key in distribution_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create the figure
    # Reverse alphas so highest alpha is at top
    alphas_reversed = list(reversed(alphas))
    fig, axes = plt.subplots(len(alphas), len(mus), 
                            figsize=(3*len(mus), 2.5*len(alphas)))
    
    # Handle case where we only have one row or column
    if len(alphas) == 1 and len(mus) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(mus) == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each distribution
    for i, alpha in enumerate(alphas_reversed):  # Highest alpha at top (row 0)
        for j, mu in enumerate(mus):              # Lowest mu at left (column 0)
            ax = axes[i][j]
            
            if (alpha, mu) in distribution_data:
                data = distribution_data[(alpha, mu)]
                ones_counts = data['ones_counts']
                
                # Create histogram
                bins = np.arange(0, B+2) - 0.5  # Bins centered on integers
                ax.hist(ones_counts, bins=bins, density=True, alpha=0.7, 
                       color='steelblue', edgecolor='black', linewidth=0.5)
                
                # Statistics
                mean_ones = np.mean(ones_counts)
                std_ones = np.std(ones_counts)
                
                ax.set_title(f'α={alpha:.2f}, μ={mu:.3f}\nμ={mean_ones:.2f}, σ={std_ones:.2f}', 
                           fontsize=8)
                ax.set_xlim(-0.5, B+0.5)
                ax.set_xticks(range(0, B+1, max(1, B//4)))
                
                # Only show y-axis labels on leftmost plots
                if j == 0:
                    ax.set_ylabel('Density', fontsize=8)
                else:
                    ax.set_yticklabels([])
                
                # Only show x-axis labels on bottom plots
                if i == len(alphas_reversed) - 1:
                    ax.set_xlabel('Number of 1s', fontsize=8)
                else:
                    ax.set_xticklabels([])
                
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'α={alpha:.2f}, μ={mu:.3f}', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall labels
    fig.suptitle(f'Distribution of 1s in Bitstrings: Alpha vs Mu Raster (All Time Steps)\n(N={N}, B={B}, γ={gamma})', 
                fontsize=16)
    fig.text(0.5, 0.02, 'Mu (Mutation Rate)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', 
             rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # Save the plot
    output_dir = "src/understandabilityVsHammingDense/plots/ones_fraction"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/ones_distribution_grid_alltimesteps_N_{N}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Distribution grid saved to: {fname}")


def main(N=1000, B=16, gamma=1):
    """Main function that generates both visualizations."""
    print(f"Loading language data from all time steps for parameters: N={N}, B={B}, gamma={gamma}")
    
    # Load data and compute both average 1s counts and distributions
    ones_data, distribution_data = load_meanfield_data(N, B, gamma)
    
    if not ones_data and not distribution_data:
        print("No valid data found.")
        return
    
    print(f"Successfully processed {len(ones_data)} parameter combinations")
    
    # Create both visualizations
    if ones_data:
        create_ones_heatmap(ones_data, N, B, gamma)
    
    if distribution_data:
        create_distribution_grid(distribution_data, N, B, gamma)

if __name__ == "__main__":
    main(N=1000, B=16, gamma=1)