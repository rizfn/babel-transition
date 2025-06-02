import numpy as np
import matplotlib.pyplot as plt
import umap
import networkx as nx
from sklearn.metrics import pairwise_distances
import os

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

def find_clique_clusters(distance_matrix, threshold):
    """
    Find clusters where all points in a cluster are within threshold distance of each other.
    Args:
        distance_matrix: Matrix of pairwise distances
        threshold: Maximum distance for points to be considered in the same cluster
    Returns:
        List of cluster labels, -1 indicates noise points
    """
    n = distance_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] <= threshold:
                G.add_edge(i, j)
    cliques = list(nx.find_cliques(G))
    cliques.sort(key=len, reverse=True)
    labels = np.full(n, -1)
    for i, clique in enumerate(cliques):
        for point in clique:
            if labels[point] == -1:
                labels[point] = i
    return labels

def plot_lattice_clique_clusters(L, gamma, alpha, mu, K, meaningful_diff_bits):
    lattice_file = f"src/simplicityVsHamming2D/outputs/lattice/L_{L}_g_{gamma}_a_{alpha}_mu_{mu}_K_{K}.tsv"
    if not os.path.exists(lattice_file):
        print(f"File not found: {lattice_file}")
        return
    lattice = load_lattice(lattice_file)
    Lval, _, B = lattice.shape
    flat = lattice.reshape(-1, B)
    # Hamming distance matrix
    hamming_matrix = pairwise_distances(flat, metric='hamming') * B
    # Clique-based clustering
    labels = find_clique_clusters(hamming_matrix, meaningful_diff_bits)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_counts = {l: np.sum(labels == l) for l in set(labels)}
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # UMAP for visualization only
    n_neighbors = min(15, len(flat)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                        metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)

    # 1. UMAP scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_umap = axes[1]
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax_umap.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                        color=color_mapping[-1], alpha=0.5, s=30)
    for label in sorted_clusters:
        mask = labels == label
        ax_umap.scatter(embedding[mask, 0], embedding[mask, 1],
                        color=color_mapping[label], alpha=0.8, s=50)
    ax_umap.set_title(f'UMAP Visualization with Clique-based Clustering\n{n_clusters} clusters (threshold={meaningful_diff_bits} bits)')
    ax_umap.set_xlabel('UMAP Dimension 1')
    ax_umap.set_ylabel('UMAP Dimension 2')

    # 2. Heatmap of languages sorted by cluster
    ax_heat = axes[0]
    sorted_indices = np.argsort(labels)
    sorted_languages = flat[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), B, 3))
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    ax_heat.imshow(language_colors, aspect='auto', interpolation='none',
                   extent=[-0.5, B-0.5, len(sorted_languages)-0.5, -0.5])
    ax_heat.set_title(f'Languages Heatmap (sorted by cluster)')
    ax_heat.set_xlabel('Bit Position')
    ax_heat.set_ylabel('Agent ID (sorted)')

    # 3. Lattice colored by cluster
    ax_lat = axes[2]
    lattice_labels = labels.reshape(Lval, Lval)
    cmap = plt.get_cmap('tab20', n_clusters)
    im = ax_lat.imshow(lattice_labels, cmap=cmap, interpolation='nearest')
    ax_lat.set_title(f"Lattice: {n_clusters} clusters")
    ax_lat.set_xlabel("X")
    ax_lat.set_ylabel("Y")
    plt.colorbar(im, ax=ax_lat, ticks=range(n_clusters), label="Cluster ID")

    outname = f"src/simplicityVsHamming2D/plots/lattice/clique_clusters_L_{L}_g_{gamma}_a_{alpha}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {outname}")
    plt.show()

if __name__ == "__main__":
    plot_lattice_clique_clusters(100, 1, 0, 0.01, 3, 5)