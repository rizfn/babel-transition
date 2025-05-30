import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import re
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, and max_depth from filename."""
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

def compute_clusters(languages, method='umap'):
    if len(languages) < 2:
        return 0, 0
    L = languages[0].shape[0]
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    avg_hamming = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))
    if method == 'umap':
        language_tuples = [tuple(lang) for lang in languages]
        unique_languages = set(language_tuples)
        diversity = len(unique_languages)
        diversity_ratio = diversity / len(languages)
        n_neighbors = min(15, len(languages)-1)
        try:
            reducer = umap.UMAP(n_components=2, 
                               n_neighbors=n_neighbors, 
                               min_dist=0.01,
                               metric='precomputed', 
                               random_state=42)
            embedding = reducer.fit_transform(hamming_matrix)
            umap_distances = pairwise_distances(embedding, metric='euclidean')
            base_eps = 0.5
            max_eps = 1.0
            scaled_eps = min(max_eps, base_eps + (1 - diversity_ratio))
            dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
            labels = dbscan.fit_predict(umap_distances)
        except Exception as e:
            print(f"Error in UMAP clustering: {e}")
            return 0, avg_hamming
    else:
        meaningful_diff_bits = 1
        hamming_eps = meaningful_diff_bits / L
        try:
            dbscan = DBSCAN(eps=hamming_eps, min_samples=1, metric='precomputed')
            labels = dbscan.fit_predict(hamming_matrix / L)
        except Exception as e:
            print(f"Error in Hamming clustering: {e}")
            return 0, avg_hamming
    n_clusters = len(set([l for l in labels if l != -1]))
    return n_clusters, avg_hamming

def main():
    # Path to language files (update to new directory and pattern)
    languages_files_pattern = "src/simpleForRelatives/outputs/beta/languages/g_*_a_*_gdmax_*_N_*_b_*_L_*_mu_*.tsv"
    language_files = glob.glob(languages_files_pattern)
    if not language_files:
        print(f"No files found matching pattern: {languages_files_pattern}")
        return
    print(f"Found {len(language_files)} language files")
    results = []
    for i, filename in tqdm(enumerate(language_files)):
        print(f"Processing file {i+1}/{len(language_files)}: {os.path.basename(filename)}")
        gamma, alpha, max_depth, N, beta, L, mu = extract_params_from_filename(filename)
        if gamma is not None and alpha is not None and max_depth is not None:
            languages = load_languages(filename)
            if len(languages) > 0:
                n_clusters, avg_hamming = compute_clusters(languages, method='umap')
                results.append((gamma, alpha, max_depth, n_clusters, avg_hamming, N, beta, L, mu))
                print(f"  Found {n_clusters} clusters, avg Hamming distance: {avg_hamming:.2f}")
    results = np.array(results)
    if len(results) == 0:
        print("No valid results found.")
        return

    # Assume all files have the same max_depth, N, beta, L, mu
    unique_max_depths = np.unique(results[:,2])
    if len(unique_max_depths) > 1:
        print("Warning: Multiple max_depth values found. Using the first one.")
    max_depth = unique_max_depths[0]
    N = results[0,5]
    beta = results[0,6]
    L = results[0,7]
    mu = results[0,8]

    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))
    clusters_grid = np.zeros((len(alphas), len(gammas)))
    clusters_grid.fill(np.nan)
    hamming_grid = np.zeros((len(alphas), len(gammas)))
    hamming_grid.fill(np.nan)
    for gamma, alpha, _, n_clusters, avg_hamming, *_ in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        clusters_grid[alpha_idx, gamma_idx] = n_clusters
        hamming_grid[alpha_idx, gamma_idx] = avg_hamming

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # 1. Number of clusters heatmap
    im1 = ax1.imshow(clusters_grid, cmap='viridis', aspect='auto',
                    extent=[min(gammas)-0.5, max(gammas)+0.5, min(alphas)-0.5, max(alphas)+0.5],
                    origin='lower')
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(clusters_grid[i, j]):
                value = clusters_grid[i, j]
                vmin, vmax = np.nanmin(clusters_grid), np.nanmax(clusters_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax1.text(gamma, alpha, f"{int(value)}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Number of Clusters', rotation=270, labelpad=20)
    ax1.set_xticks(gammas)
    ax1.set_yticks(alphas)
    ax1.set_xticks(gammas, minor=True)
    ax1.set_yticks(alphas, minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Gamma (hamming distance penalty)')
    ax1.set_ylabel('Alpha (simplicity bonus)')
    ax1.set_title('Number of Language Clusters Across (Alpha, Gamma)')

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
                ax2.text(gamma, alpha, f"{value:.2f}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance', rotation=270, labelpad=20)
    ax2.set_xticks(gammas)
    ax2.set_yticks(alphas)
    ax2.set_xticks(gammas, minor=True)
    ax2.set_yticks(alphas, minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Gamma (hamming distance penalty)')
    ax2.set_ylabel('Alpha (simplicity bonus)')
    ax2.set_title('Average Hamming Distance Across (Alpha, Gamma)')

    output_dir = "src/simpleForRelatives/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    fname = (f"{output_dir}/beta_{beta}_heatmap_N_{int(N)}_L_{int(L)}_gdmax_{int(max_depth)}_mu_{mu}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
