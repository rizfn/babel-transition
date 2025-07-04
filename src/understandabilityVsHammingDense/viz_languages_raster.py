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
    """Load languages from a TSV file and return as binary arrays."""
    try:
        df = pd.read_csv(filename, sep='\t', dtype={'language': str})
        if 'generation' in df.columns:
            last_gen = df['generation'].max()
            df = df[df['generation'] == last_gen]
        
        languages = []
        for lang_str in df['language']:
            lang_str = str(lang_str)
            lang_array = np.array([int(bit) for bit in lang_str])
            languages.append(lang_array)
        return np.array(languages)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return np.array([])

def compute_language_diversity(languages):
    """Compute number of unique languages and average Hamming distance."""
    if len(languages) < 2:
        return 0, 0
    
    # Count unique languages
    language_tuples = [tuple(lang) for lang in languages]
    unique_languages = set(language_tuples)
    n_unique = len(unique_languages)
    
    # Compute average Hamming distance
    B = languages[0].shape[0]
    total_hamming = 0
    n_comparisons = 0
    
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            hamming_dist = np.sum(languages[i] != languages[j])
            total_hamming += hamming_dist
            n_comparisons += 1
    
    avg_hamming = total_hamming / n_comparisons if n_comparisons > 0 else 0
    
    return n_unique, avg_hamming

def main(N, B, mu):
    """Main function to process language files and create heatmaps."""
    # Path to language files
    base_dir = os.path.dirname(__file__)
    languages_files_pattern = os.path.join(base_dir, f"outputs/top50/languages/g_*_a_*_N_{N}_B_{B}_mu_{mu}.tsv")
    language_files = glob.glob(languages_files_pattern)
    
    if not language_files:
        print(f"No files found matching pattern: {languages_files_pattern}")
        return
    
    print(f"Found {len(language_files)} language files")
    
    results = []
    for i, filename in tqdm(enumerate(language_files), desc="Processing files"):
        print(f"Processing file {i+1}/{len(language_files)}: {os.path.basename(filename)}")
        gamma, alpha, N_file, B_file, mu_file = extract_params_from_filename(filename)
        
        if gamma is not None and alpha is not None:
            languages = load_languages(filename)
            if len(languages) > 0:
                n_unique, avg_hamming = compute_language_diversity(languages)
                results.append((gamma, alpha, n_unique, avg_hamming, N_file, B_file, mu_file))
                print(f"  Found {n_unique} unique languages, avg Hamming distance: {avg_hamming:.2f}")

    results = np.array(results)
    if len(results) == 0:
        print("No valid results found.")
        return

    # Extract parameter ranges
    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))
    N = int(results[0, 4])
    B = int(results[0, 5])
    mu = results[0, 6]

    # Create grids for heatmaps
    unique_grid = np.zeros((len(alphas), len(gammas)))
    unique_grid.fill(np.nan)
    hamming_grid = np.zeros((len(alphas), len(gammas)))
    hamming_grid.fill(np.nan)

    for gamma, alpha, n_unique, avg_hamming, *_ in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        unique_grid[alpha_idx, gamma_idx] = n_unique
        hamming_grid[alpha_idx, gamma_idx] = avg_hamming

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Number of unique languages heatmap
    im1 = ax1.imshow(unique_grid, cmap='viridis', aspect='auto',
                    extent=[min(gammas)-0.5, max(gammas)+0.5, min(alphas)-0.5, max(alphas)+0.5],
                    origin='lower')

    # Add text annotations
    x_centers = np.linspace(min(gammas), max(gammas), len(gammas))
    y_centers = np.linspace(min(alphas), max(alphas), len(alphas))
    
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(unique_grid[i, j]):
                value = unique_grid[i, j]
                vmin, vmax = np.nanmin(unique_grid), np.nanmax(unique_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax1.text(x_centers[j], y_centers[i], f"{int(value)}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Number of Unique Languages', rotation=270, labelpad=20)
    ax1.set_xticks(gammas)
    ax1.set_yticks(alphas)
    ax1.set_xticks(gammas, minor=True)
    ax1.set_yticks(alphas, minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Gamma (Global Interaction Strength)')
    ax1.set_ylabel('Alpha (Local Interaction Strength)')
    ax1.set_title('Number of Unique Languages')

    # 2. Average Hamming distance heatmap
    im2 = ax2.imshow(hamming_grid, cmap='plasma', aspect='auto',
                    extent=[min(gammas)-0.5, max(gammas)+0.5, min(alphas)-0.5, max(alphas)+0.5],
                    origin='lower')

    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(hamming_grid[i, j]):
                value = hamming_grid[i, j]
                vmin, vmax = np.nanmin(hamming_grid), np.nanmax(hamming_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax2.text(x_centers[j], y_centers[i], f"{value:.2f}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance', rotation=270, labelpad=20)
    ax2.set_xticks(gammas)
    ax2.set_yticks(alphas)
    ax2.set_xticks(gammas, minor=True)
    ax2.set_yticks(alphas, minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Gamma (Global Interaction Strength)')
    ax2.set_ylabel('Alpha (Local Interaction Strength)')
    ax2.set_title('Average Hamming Distance')

    # Overall title
    fig.suptitle(f'Language Diversity Analysis (N={N}, B={B}, μ={mu})', fontsize=16)

    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), "plots/languages")
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/heatmap_top50_N_{N}_B_{B}_mu_{mu}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')

    print(f"Plot saved to: {fname}")

if __name__ == "__main__":
    # Set your parameters here
    main(N=1000, B=16, mu=0.01)