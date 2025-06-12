# import numpy as np
# import matplotlib.pyplot as plt
# import umap
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import pairwise_distances
# from scipy.optimize import linear_sum_assignment
# import os

# def parse_lattice_line(line, L, B):
#     """Parse a single line from the new lattice timeseries format."""
#     step, lattice_str = line.strip().split('\t')
#     rows = lattice_str.split(';')
#     lattice = np.zeros((L, L, B), dtype=int)
#     for i, row in enumerate(rows):
#         cells = row.split(',')
#         for j, cell in enumerate(cells):
#             bits = [int(b) for b in cell]
#             lattice[i, j, :] = bits
#     return int(step), lattice

# def cluster_umap_lattice(lattice_3d):
#     """Flatten lattice to (N, B), run UMAP, cluster, and return labels and embedding."""
#     L, _, B = lattice_3d.shape
#     flat = lattice_3d.reshape(-1, B)
#     # Hamming matrix
#     hamming_matrix = pairwise_distances(flat, metric='hamming') * B
#     # UMAP
#     n_neighbors = min(15, len(flat)-1)
#     reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
#                         metric='precomputed', random_state=42)
#     embedding = reducer.fit_transform(hamming_matrix)
#     # DBSCAN on UMAP
#     umap_distances = pairwise_distances(embedding, metric='euclidean')
#     diversity = len(set(tuple(row) for row in flat)) / len(flat)
#     standard_eps = 1
#     scaled_eps = standard_eps * (1 - diversity)
#     dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
#     labels = dbscan.fit_predict(umap_distances)
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     return labels, embedding, n_clusters

# def compute_lattice_centroids(labels, L):
#     """Compute centroids of clusters in lattice coordinates."""
#     centroids = {}
#     lattice_labels = labels.reshape(L, L)
#     for label in set(labels):
#         if label == -1:
#             continue
#         positions = np.argwhere(lattice_labels == label)
#         centroids[label] = positions.mean(axis=0)
#     return centroids

# def match_clusters(prev_centroids, curr_centroids, threshold):
#     """Match clusters by minimizing centroid distance."""
#     prev_labels = list(prev_centroids.keys())
#     curr_labels = list(curr_centroids.keys())
#     if not prev_labels or not curr_labels:
#         return {}, set(curr_labels)
#     cost = np.zeros((len(curr_labels), len(prev_labels)))
#     for i, cl in enumerate(curr_labels):
#         for j, pl in enumerate(prev_labels):
#             cost[i, j] = np.linalg.norm(curr_centroids[cl] - prev_centroids[pl])
#     row_ind, col_ind = linear_sum_assignment(cost)
#     mapping = {}
#     unmatched = set(curr_labels)
#     for r, c in zip(row_ind, col_ind):
#         if cost[r, c] < threshold:
#             mapping[curr_labels[r]] = prev_labels[c]
#             unmatched.discard(curr_labels[r])
#     return mapping, unmatched

# def plot_lattice_umap_clusters_frame(lattice, labels, embedding, n_clusters, step, outdir, color_mapping):
#     Lval, _, B = lattice.shape
#     flat = lattice.reshape(-1, B)
#     cluster_counts = {l: np.sum(labels == l) for l in set(labels)}
#     sorted_clusters = sorted([l for l in set(labels) if l != -1],
#                              key=lambda l: -cluster_counts[l])

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     # 1. UMAP scatter
#     ax_umap = axes[1]
#     noise_mask = labels == -1
#     if np.any(noise_mask):
#         ax_umap.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
#                         color=color_mapping[-1], alpha=0.5, s=30)
#     for label in sorted_clusters:
#         mask = labels == label
#         ax_umap.scatter(embedding[mask, 0], embedding[mask, 1],
#                         color=color_mapping[label], alpha=0.8, s=50)
#     ax_umap.set_title(f'UMAP+DBSCAN Clustering\n{n_clusters} clusters')
#     ax_umap.set_xlabel('UMAP Dimension 1')
#     ax_umap.set_ylabel('UMAP Dimension 2')

#     # 2. Heatmap of languages sorted by cluster
#     ax_heat = axes[0]
#     sorted_indices = np.argsort(labels)
#     sorted_languages = flat[sorted_indices]
#     sorted_labels = labels[sorted_indices]
#     language_colors = np.ones((len(sorted_languages), B, 3))
#     for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
#         color = color_mapping[label]
#         for j, bit in enumerate(language):
#             if bit == 1:
#                 language_colors[i, j] = color
#     ax_heat.imshow(language_colors, aspect='auto', interpolation='none',
#                    extent=[-0.5, B-0.5, len(sorted_languages)-0.5, -0.5])
#     ax_heat.set_title(f'Languages Heatmap (sorted by cluster)')
#     ax_heat.set_xlabel('Bit Position')
#     ax_heat.set_ylabel('Agent ID (sorted)')

#     # 3. Lattice colored by cluster
#     ax_lat = axes[2]
#     lattice_labels = labels.reshape(Lval, Lval)
#     lattice_rgb = np.ones((Lval, Lval, 3))
#     for i in range(Lval):
#         for j in range(Lval):
#             lattice_rgb[i, j] = color_mapping[lattice_labels[i, j]]
#     ax_lat.imshow(lattice_rgb, interpolation='nearest')
#     ax_lat.set_title(f"Lattice: {n_clusters} clusters")
#     ax_lat.set_xlabel("X")
#     ax_lat.set_ylabel("Y")

#     plt.suptitle(f"Step {step}")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     os.makedirs(outdir, exist_ok=True)
#     outname = os.path.join(outdir, f"frame_{step:04d}.png")
#     plt.savefig(outname, dpi=150, bbox_inches='tight')
#     plt.close()

# def process_lattice_timeseries(L, B, gamma, alpha, r, mu, K, cluster_threshold):
#     filename = f"src/understandabilityVsHamming2D/outputs/latticeNbrVsGlobalTimeseries/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}.tsv"
#     outdir = f"src/understandabilityVsHamming2D/plots/latticeAnimNbrVsGlobal/frames/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}"
#     prev_centroids = {}
#     prev_color_mapping = {}
#     color_pool = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
#     color_idx = 0

#     with open(filename, "r") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             step, lattice = parse_lattice_line(line, L, B)
#             labels, embedding, n_clusters = cluster_umap_lattice(lattice)
#             curr_centroids = compute_lattice_centroids(labels, L)
#             mapping, unmatched = match_clusters(prev_centroids, curr_centroids, cluster_threshold)
#             color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
#             # Assign colors: matched clusters get previous color, new clusters get new color
#             for cl in curr_centroids:
#                 if cl in mapping and mapping[cl] in prev_color_mapping:
#                     color_mapping[cl] = prev_color_mapping[mapping[cl]]
#                 else:
#                     color_mapping[cl] = np.array(color_pool[color_idx % len(color_pool)])
#                     color_idx += 1
#             plot_lattice_umap_clusters_frame(
#                 lattice, labels, embedding, n_clusters, step, outdir, color_mapping
#             )
#             prev_centroids = curr_centroids
#             prev_color_mapping = color_mapping

# if __name__ == "__main__":
#     L = 128
#     B = 32
#     gamma = 2
#     alpha = 1
#     r = 0.5
#     mu = 0.01
#     K = 3

#     cluster_threshold = 16
#     process_lattice_timeseries(L, B, gamma, alpha, r, mu, K, cluster_threshold)


import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import os
from collections import defaultdict

def parse_lattice_line(line, L, B):
    """Parse a single line from the new lattice timeseries format."""
    step, lattice_str = line.strip().split('\t')
    rows = lattice_str.split(';')
    lattice = np.zeros((L, L, B), dtype=int)
    for i, row in enumerate(rows):
        cells = row.split(',')
        for j, cell in enumerate(cells):
            bits = [int(b) for b in cell]
            lattice[i, j, :] = bits
    return int(step), lattice

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

def match_clusters_overlap(prev_labels, curr_labels, overlap_threshold=0.3):
    """
    Match clusters by maximizing overlap of lattice points.
    prev_labels, curr_labels: 2D arrays of shape (L, L)
    Returns: mapping (curr_label -> prev_label), unmatched (set of curr_labels)
    """
    prev_label_set = set(np.unique(prev_labels)) - {-1}
    curr_label_set = set(np.unique(curr_labels)) - {-1}
    mapping = {}
    unmatched = set(curr_label_set)
    prev_label_counts = {pl: np.sum(prev_labels == pl) for pl in prev_label_set}
    used_prev_labels = set()

    for cl in curr_label_set:
        overlap_counts = defaultdict(int)
        mask = (curr_labels == cl)
        prev_labels_in_mask = prev_labels[mask]
        for pl in prev_label_set:
            overlap_counts[pl] = np.sum(prev_labels_in_mask == pl)
        if overlap_counts:
            # Find the previous label with the most overlap
            best_prev = max(overlap_counts, key=overlap_counts.get)
            overlap_frac = overlap_counts[best_prev] / np.sum(mask)
            # Only assign if overlap is significant and not already used (handles splits)
            if overlap_frac > overlap_threshold and best_prev not in used_prev_labels:
                mapping[cl] = best_prev
                unmatched.discard(cl)
                used_prev_labels.add(best_prev)
    return mapping, unmatched

def plot_lattice_umap_clusters_frame(lattice, labels, embedding, n_clusters, step, outdir, color_mapping):
    Lval, _, B = lattice.shape
    flat = lattice.reshape(-1, B)
    cluster_counts = {l: np.sum(labels == l) for l in set(labels)}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 1. UMAP scatter
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
    lattice_rgb = np.ones((Lval, Lval, 3))
    for i in range(Lval):
        for j in range(Lval):
            lattice_rgb[i, j] = color_mapping[lattice_labels[i, j]]
    ax_lat.imshow(lattice_rgb, interpolation='nearest')
    ax_lat.set_title(f"Lattice: {n_clusters} clusters")
    ax_lat.set_xlabel("X")
    ax_lat.set_ylabel("Y")

    plt.suptitle(f"Step {step}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, f"frame_{step:04d}.png")
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()

def process_lattice_timeseries(L, B, gamma, alpha, r, mu, K, overlap_threshold=0.3):
    filename = f"src/understandabilityVsHamming2D/outputs/latticeNbrVsGlobalTimeseries/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}.tsv"
    outdir = f"src/understandabilityVsHamming2D/plots/latticeAnimNbrVsGlobal/frames/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}"
    prev_labels_2d = None
    prev_color_mapping = {}
    color_pool = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_idx = 0

    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            labels, embedding, n_clusters = cluster_umap_lattice(lattice)
            labels_2d = labels.reshape(L, L)
            color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
            if prev_labels_2d is not None:
                mapping, unmatched = match_clusters_overlap(prev_labels_2d, labels_2d, overlap_threshold)
            else:
                mapping, unmatched = {}, set(np.unique(labels)) - {-1}
            for cl in np.unique(labels):
                if cl == -1:
                    continue
                if cl in mapping and mapping[cl] in prev_color_mapping:
                    color_mapping[cl] = prev_color_mapping[mapping[cl]]
                else:
                    color_mapping[cl] = np.array(color_pool[color_idx % len(color_pool)])
                    color_idx += 1
            plot_lattice_umap_clusters_frame(
                lattice, labels, embedding, n_clusters, step, outdir, color_mapping
            )
            prev_labels_2d = labels_2d
            prev_color_mapping = color_mapping

if __name__ == "__main__":
    L = 128
    B = 32
    gamma = 2
    alpha = 1
    r = 2
    mu = 0.01
    K = 3
    overlap_threshold = 0.3  # Fraction of overlap required to consider clusters the same
    process_lattice_timeseries(L, B, gamma, alpha, r, mu, K, overlap_threshold)