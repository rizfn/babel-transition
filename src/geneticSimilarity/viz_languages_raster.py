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
    """Extract gamma and max_depth from filename."""
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    max_depth_match = re.search(r'gdmax_([0-9]+)', filename)
    if gamma_match and max_depth_match:
        gamma = float(gamma_match.group(1))
        max_depth = int(max_depth_match.group(1))
        return gamma, max_depth
    else:
        return None, None

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
    languages_files_pattern = "src/geneticSimilarity/outputs/betaNegative/languages/g_*_N_*_L_*_mu_*_gdmax_*.tsv"
    language_files = glob.glob(languages_files_pattern)
    if not language_files:
        print(f"No files found matching pattern: {languages_files_pattern}")
        return
    print(f"Found {len(language_files)} language files")
    results = []
    for i, filename in tqdm(enumerate(language_files)):
        print(f"Processing file {i+1}/{len(language_files)}: {os.path.basename(filename)}")
        gamma, max_depth = extract_params_from_filename(filename)
        if gamma is not None and max_depth is not None:
            languages = load_languages(filename)
            if len(languages) > 0:
                n_clusters, avg_hamming = compute_clusters(languages, method='umap')
                results.append((gamma, max_depth, n_clusters, avg_hamming))
                print(f"  Found {n_clusters} clusters, avg Hamming distance: {avg_hamming:.2f}")
    results = np.array(results)
    if len(results) == 0:
        print("No valid results found.")
        return
    gammas = np.sort(np.unique(results[:, 0]))
    max_depths = np.sort(np.unique(results[:, 1]))
    clusters_grid = np.zeros((len(gammas), len(max_depths)))
    clusters_grid.fill(np.nan)
    hamming_grid = np.zeros((len(gammas), len(max_depths)))
    hamming_grid.fill(np.nan)
    for gamma, max_depth, n_clusters, avg_hamming in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        max_depth_idx = np.where(max_depths == max_depth)[0][0]
        clusters_grid[gamma_idx, max_depth_idx] = n_clusters
        hamming_grid[gamma_idx, max_depth_idx] = avg_hamming
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # 1. Number of clusters heatmap
    im1 = ax1.imshow(clusters_grid, cmap='viridis', aspect='auto',
                    extent=[min(max_depths)-0.5, max(max_depths)+0.5, min(gammas)-0.5, max(gammas)+0.5],
                    origin='lower')
    for i, gamma in enumerate(gammas):
        for j, max_depth in enumerate(max_depths):
            if not np.isnan(clusters_grid[i, j]):
                value = clusters_grid[i, j]
                vmin, vmax = np.nanmin(clusters_grid), np.nanmax(clusters_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax1.text(max_depth, gamma, f"{int(value)}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Number of Clusters', rotation=270, labelpad=20)
    ax1.set_xticks(max_depths)
    ax1.set_yticks(gammas)
    ax1.set_xticks(max_depths, minor=True)
    ax1.set_yticks(gammas, minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('max_depth (gdmax)')
    ax1.set_ylabel('Gamma (hamming distance penalty)')
    ax1.set_title('Number of Language Clusters Across Parameter Space')
    # 2. Average Hamming distance heatmap
    im2 = ax2.imshow(hamming_grid, cmap='plasma', aspect='auto',
                    extent=[min(max_depths)-0.5, max(max_depths)+0.5, min(gammas)-0.5, max(gammas)+0.5],
                    origin='lower')
    for i, gamma in enumerate(gammas):
        for j, max_depth in enumerate(max_depths):
            if not np.isnan(hamming_grid[i, j]):
                value = hamming_grid[i, j]
                vmin, vmax = np.nanmin(hamming_grid), np.nanmax(hamming_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax2.text(max_depth, gamma, f"{value:.2f}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance', rotation=270, labelpad=20)
    ax2.set_xticks(max_depths)
    ax2.set_yticks(gammas)
    ax2.set_xticks(max_depths, minor=True)
    ax2.set_yticks(gammas, minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('max_depth (gdmax)')
    ax2.set_ylabel('Gamma (hamming distance penalty)')
    ax2.set_title('Average Hamming Distance Across Parameter Space')
    output_dir = "src/geneticSimilarity/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/betaNegative_heatmap.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()