import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
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

def cluster_umap_lattice(lattice_3d):
    """Flatten lattice to (N, B), run UMAP, cluster, and return labels and embedding."""
    L, _, B = lattice_3d.shape
    flat = lattice_3d.reshape(-1, B)
    # Hamming matrix
    hamming_matrix = pairwise_distances(flat, metric='hamming') * B
    # UMAP
    n_neighbors = min(15, len(flat)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                        metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)
    # DBSCAN on UMAP
    umap_distances = pairwise_distances(embedding, metric='euclidean')
    diversity = len(set(tuple(row) for row in flat)) / len(flat)
    standard_eps = 1
    scaled_eps = standard_eps * (1 - diversity)
    dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(umap_distances)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, embedding, n_clusters

def plot_lattice_umap_clusters_neighbours(L, gamma, alpha, globalInteractionRatio, mu, K):
    lattice_file = f"src/simpleForRelatives2D/outputs/lattice/neighbours/L_{L}_g_{gamma}_a_{alpha}_r_{globalInteractionRatio}_mu_{mu}_K_{K}.tsv"
    if not os.path.exists(lattice_file):
        print(f"File not found: {lattice_file}")
        return
    lattice = load_lattice(lattice_file)
    Lval, _, B = lattice.shape
    flat = lattice.reshape(-1, B)
    labels, embedding, n_clusters = cluster_umap_lattice(lattice)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    cluster_counts = {l: np.sum(labels == l) for l in set(labels)}
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

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
    ax_umap.set_title(f'UMAP+DBSCAN Clustering\n{n_clusters} clusters')
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

    outname = f"src/simpleForRelatives2D/plots/lattice/neighbours/umap_clusters_L_{L}_g_{gamma}_r_{globalInteractionRatio}_a_{alpha}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lattice_umap_clusters_neighbours(L, gamma, alpha, globalInteractionRatio, mu, K):
    lattice_file = f"src/simpleForRelatives2D/outputs/lattice/geneticDist/L_{L}_g_{gamma}_a_{alpha}_r_{globalInteractionRatio}_mu_{mu}_K_{K}.tsv"
    if not os.path.exists(lattice_file):
        print(f"File not found: {lattice_file}")
        return
    lattice = load_lattice(lattice_file)
    Lval, _, B = lattice.shape
    flat = lattice.reshape(-1, B)
    labels, embedding, n_clusters = cluster_umap_lattice(lattice)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    cluster_counts = {l: np.sum(labels == l) for l in set(labels)}
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

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
    ax_umap.set_title(f'UMAP+DBSCAN Clustering\n{n_clusters} clusters')
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

    outname = f"src/simpleForRelatives2D/plots/lattice/geneticDist/umap_clusters_L_{L}_g_{gamma}_r_{globalInteractionRatio}_a_{alpha}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    L = 100
    gamma = 2
    alpha = 2
    globalInteractionRatio = 2
    mu = 0.01
    K = 3

    plot_lattice_umap_clusters_neighbours(L, gamma, alpha, globalInteractionRatio, mu, K)