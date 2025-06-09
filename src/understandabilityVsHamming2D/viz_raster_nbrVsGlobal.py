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
    """Extract gamma, alpha, gir, L, B, mu, K from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    gir = re.search(r'r_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9.]+)', filename)
    K = re.search(r'K_([0-9]+)', filename)
    if gamma and alpha and gir and L and B and mu and K:
        return (float(gamma.group(1)), float(alpha.group(1)), float(gir.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)), int(K.group(1)))
    return (None,)*7

def load_lattice(filename):
    """Load a lattice file (TSV) into a 3D numpy array: shape (L, L, B)."""
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    lattice = []
    for line in lines:
        row = []
        for cell in line.split('\t'):
            bits = [int(b) for b in cell]
            row.append(bits)
        lattice.append(row)
    arr = np.array(lattice)  # shape (L, L, B)
    return arr

def compute_clusters(lattice_3d):
    L, _, B = lattice_3d.shape
    flat = lattice_3d.reshape(-1, B)
    hamming_matrix = pairwise_distances(flat, metric='hamming') * B
    avg_hamming = np.sum(hamming_matrix) / (len(flat) * (len(flat) - 1))
    language_tuples = [tuple(lang) for lang in flat]
    unique_languages = set(language_tuples)
    diversity = len(unique_languages)
    diversity_ratio = diversity / len(flat)
    n_neighbors = min(15, len(flat)-1)
    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                            metric='precomputed', random_state=42)
        embedding = reducer.fit_transform(hamming_matrix)
        umap_distances = pairwise_distances(embedding, metric='euclidean')
        standard_eps = 5
        scaled_eps = standard_eps * (1 - diversity_ratio)
        dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(umap_distances)
    except Exception as e:
        print(f"Error in UMAP clustering: {e}")
        return 0, avg_hamming
    n_clusters = len(set([l for l in labels if l != -1]))
    return n_clusters, avg_hamming

def main_latticeNbrVsGlobal(gir_filter):
    # Path to lattice files
    pattern = os.path.join(os.path.dirname(__file__), "outputs/latticeNbrVsGlobal/L_*_g_*_a_*_r_{}_B_*_mu_*_K_*.tsv".format(gir_filter))
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for gir={gir_filter} with pattern: {pattern}")
        return
    print(f"Found {len(files)} files for gir={gir_filter}")
    results = []
    for filename in tqdm(files):
        gamma, alpha, gir, L, B, mu, K = extract_params_from_filename(filename)
        if gamma is not None and alpha is not None:
            lattice = load_lattice(filename)
            n_clusters, avg_hamming = compute_clusters(lattice)
            results.append((gamma, alpha, n_clusters, avg_hamming, L, B, mu, gir, K))
    results = np.array(results)
    if len(results) == 0:
        print("No valid results found.")
        return

    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))

    clusters_grid = np.full((len(alphas), len(gammas)), np.nan)
    hamming_grid = np.full((len(alphas), len(gammas)), np.nan)
    for gamma, alpha, n_clusters, avg_hamming, *_ in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        clusters_grid[alpha_idx, gamma_idx] = n_clusters
        hamming_grid[alpha_idx, gamma_idx] = avg_hamming

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # 1. Number of clusters heatmap
    im1 = ax1.imshow(clusters_grid, cmap='viridis', aspect='auto',
                    extent=[min(gammas)-0.5, max(gammas)+0.5, min(alphas)-0.5, max(alphas)+0.5],
                    origin='lower')
    x_centers = np.linspace(min(gammas), max(gammas), len(gammas))
    y_centers = np.linspace(min(alphas), max(alphas), len(alphas))
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(clusters_grid[i, j]):
                value = clusters_grid[i, j]
                vmin, vmax = np.nanmin(clusters_grid), np.nanmax(clusters_grid)
                brightness = 0.5 if vmin == vmax else (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                ax1.text(x_centers[j], y_centers[i], f"{int(value)}", 
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
    ax1.set_title(f'Number of Language Clusters (gir={gir_filter})')

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
    ax2.set_xlabel('Gamma (hamming distance penalty)')
    ax2.set_ylabel('Alpha (simplicity bonus)')
    ax2.set_title(f'Average Hamming Distance (gir={gir_filter})')

    output_dir = "src/understandabilityVsHamming2D/plots/latticeNbrVsGlobal"
    os.makedirs(output_dir, exist_ok=True)
    fname = (f"{output_dir}/heatmap_latticeNbrVsGlobal_gir_{gir_filter}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_latticeNbrVsGlobal(2)