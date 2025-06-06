import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import os

def hamming(a, b):
    """Calculate the Hamming distance between two binary arrays."""
    return np.sum(a != b)

def umap_clusters(gamma, alpha, N, L, mu):
    """Run UMAP and cluster for a simplicity-vs-hamming experiment."""
    # Load language data
    base_dir = "src/understandabilityVsHamming/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Use last generation if present
    if 'generation' in df.columns:
        last_gen = df['generation'].max()
        df = df[df['generation'] == last_gen]
    else:
        last_gen = None

    # Convert language strings to binary arrays
    languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df['language']])

    # Diversity
    language_tuples = [tuple(lang) for lang in languages]
    unique_languages = set(language_tuples)
    diversity = len(unique_languages)
    diversity_ratio = diversity / len(languages)

    # Color palette
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)

    # Hamming matrix
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    avg_hamming = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))

    # UMAP
    n_neighbors = min(15, len(languages)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                        metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)

    # DBSCAN on UMAP
    umap_distances = pairwise_distances(embedding, metric='euclidean')
    standard_eps = 5
    scaled_eps = standard_eps * (1 - diversity_ratio)
    dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(umap_distances)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Cluster counts
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    # Color mapping
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # UMAP scatter
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                    color=color_mapping[-1], alpha=0.5, s=30)
    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    color=color_mapping[label], alpha=0.8, s=50)
    gen_info = f"generation {last_gen}" if last_gen is not None else "final generation"
    ax2.set_title(f'UMAP+DBSCAN Clustering\n'
                  f'{n_clusters} clusters (γ={gamma}, diversity={diversity_ratio:.2%}, eps={scaled_eps:.2f}, {gen_info})')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')

    # Heatmap
    sorted_indices = np.argsort(labels)
    sorted_languages = languages[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), L, 3))
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    ax1.imshow(language_colors, aspect='auto', interpolation='none',
               extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    ax1.set_title(f'Languages Heatmap (All {len(languages)} Languages)')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Language ID (sorted by cluster)')

    plt.tight_layout()
    output_dir = "src/understandabilityVsHamming/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    gen_suffix = f"_gen_{last_gen}" if last_gen is not None else ""
    output_file = f"{output_dir}/umap_clusters_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

def hamming_clusters(gamma, alpha, N, L, mu, hamming_eps=2):
    """
    Cluster languages directly in Hamming space using DBSCAN,
    then plot UMAP and heatmap colored by Hamming clusters.
    """
    # Load language data
    base_dir = "src/understandabilityVsHamming/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Use last generation if present
    if 'generation' in df.columns:
        last_gen = df['generation'].max()
        df = df[df['generation'] == last_gen]
    else:
        last_gen = None

    # Convert language strings to binary arrays
    languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df['language']])

    # Hamming distance matrix
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L

    # DBSCAN in Hamming space
    dbscan = DBSCAN(eps=hamming_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(hamming_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Cluster counts
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    # Color mapping
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # UMAP for visualization only
    n_neighbors = min(15, len(languages)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                        metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # UMAP scatter colored by Hamming clusters
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                    color=color_mapping[-1], alpha=0.5, s=30)
    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    color=color_mapping[label], alpha=0.8, s=50)
    gen_info = f"generation {last_gen}" if last_gen is not None else "final generation"
    ax2.set_title(f'Hamming-space DBSCAN\n'
                  f'{n_clusters} clusters (eps={hamming_eps}, {gen_info})')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')

    # Heatmap colored by Hamming clusters
    sorted_indices = np.argsort(labels)
    sorted_languages = languages[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), L, 3))
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    ax1.imshow(language_colors, aspect='auto', interpolation='none',
               extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    ax1.set_title(f'Languages Heatmap (Hamming clusters)')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Language ID (sorted by cluster)')

    plt.tight_layout()
    output_dir = "src/understandabilityVsHamming/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    gen_suffix = f"_gen_{last_gen}" if last_gen is not None else ""
    output_file = f"{output_dir}/hamming_clusters_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

def umap_clusters_twostage(gamma, alpha, N, L, mu, cluster_umap_dim=10, viz_umap_dim=2):
    """
    Run two-stage UMAP clustering:
    - First, reduce to cluster_umap_dim (e.g., 10D) and cluster in that space.
    - Then, reduce to viz_umap_dim (e.g., 2D) for visualization, coloring by cluster.
    """
    # Load language data
    base_dir = "src/understandabilityVsHamming/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Use last generation if present
    if 'generation' in df.columns:
        last_gen = df['generation'].max()
        df = df[df['generation'] == last_gen]
    else:
        last_gen = None

    # Convert language strings to binary arrays
    languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df['language']])

    # Hamming matrix
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L

    # First UMAP: cluster_umap_dim
    n_neighbors = min(15, len(languages)-1)
    reducer_cluster = umap.UMAP(n_components=cluster_umap_dim, n_neighbors=n_neighbors, min_dist=0.01,
                                metric='precomputed', random_state=42)
    embedding_cluster = reducer_cluster.fit_transform(hamming_matrix)

    # DBSCAN in cluster_umap_dim space
    umap_distances = pairwise_distances(embedding_cluster, metric='euclidean')
    # Use a scaled eps as before
    language_tuples = [tuple(lang) for lang in languages]
    unique_languages = set(language_tuples)
    diversity_ratio = len(unique_languages) / len(languages)
    standard_eps = 5
    scaled_eps = standard_eps * (1 - diversity_ratio)
    dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(umap_distances)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Cluster counts and color mapping
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # Second UMAP: viz_umap_dim for visualization
    reducer_viz = umap.UMAP(n_components=viz_umap_dim, n_neighbors=n_neighbors, min_dist=0.01,
                            metric='precomputed', random_state=42)
    embedding_viz = reducer_viz.fit_transform(hamming_matrix)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # UMAP scatter (viz_umap_dim) colored by clusters found in cluster_umap_dim
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding_viz[noise_mask, 0], embedding_viz[noise_mask, 1],
                    color=color_mapping[-1], alpha=0.5, s=30)
    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding_viz[mask, 0], embedding_viz[mask, 1],
                    color=color_mapping[label], alpha=0.8, s=50)
    gen_info = f"generation {last_gen}" if last_gen is not None else "final generation"
    ax2.set_title(f'2-stage UMAP+DBSCAN\n'
                  f'{n_clusters} clusters (eps={scaled_eps:.2f}, {gen_info})')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')

    # Heatmap colored by clusters
    sorted_indices = np.argsort(labels)
    sorted_languages = languages[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), L, 3))
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    ax1.imshow(language_colors, aspect='auto', interpolation='none',
               extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    ax1.set_title(f'Languages Heatmap (2-stage UMAP clusters)')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Language ID (sorted by cluster)')

    plt.tight_layout()
    output_dir = "src/understandabilityVsHamming/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    gen_suffix = f"_gen_{last_gen}" if last_gen is not None else ""
    output_file = f"{output_dir}/umap2stage_clusters_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances
import networkx as nx

def bron_kerbosch_clique_clusters(gamma, alpha, N, L, mu, hamming_threshold=1):
    """
    Find clusters as maximal cliques in the Hamming graph using the Bron–Kerbosch algorithm.
    Each node is a language; edges connect languages with Hamming distance <= threshold.
    """
    # Load language data
    base_dir = "src/understandabilityVsHamming/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Use last generation if present
    if 'generation' in df.columns:
        last_gen = df['generation'].max()
        df = df[df['generation'] == last_gen]
    else:
        last_gen = None

    # Convert language strings to binary arrays
    languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df['language']])
    n = len(languages)

    # Build Hamming graph
    print("Building Hamming graph...")
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_matrix[i, j] <= hamming_threshold:
                G.add_edge(i, j)

    # Find all maximal cliques using Bron–Kerbosch
    print("Finding maximal cliques (Bron–Kerbosch)...")
    cliques = list(nx.find_cliques(G))
    cliques.sort(key=len, reverse=True)

    # Assign cluster labels: each node gets the first (largest) clique it appears in
    labels = np.full(n, -1)
    for idx, clique in enumerate(cliques):
        for node in clique:
            if labels[node] == -1:
                labels[node] = idx

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    # Color mapping
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_mapping = {-1: np.array([0.8, 0.8, 0.8])}
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # UMAP for visualization
    n_neighbors = min(15, n-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                        metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # UMAP scatter colored by clique clusters
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                    color=color_mapping[-1], alpha=0.5, s=30)
    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    color=color_mapping[label], alpha=0.8, s=50)
    gen_info = f"generation {last_gen}" if last_gen is not None else "final generation"
    ax2.set_title(f'UMAP Visualization with Clique Clusters\n'
                  f'{n_clusters} cliques (threshold={hamming_threshold}, {gen_info})')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')

    # Heatmap colored by clique clusters
    sorted_indices = np.argsort(labels)
    sorted_languages = languages[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), L, 3))
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    ax1.imshow(language_colors, aspect='auto', interpolation='none',
               extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    ax1.set_title(f'Languages Heatmap (Clique clusters)')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Language ID (sorted by cluster)')

    plt.tight_layout()
    output_dir = "src/understandabilityVsHamming/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    gen_suffix = f"_gen_{last_gen}" if last_gen is not None else ""
    output_file = f"{output_dir}/bronkerbosch_cliques_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}_thresh_{hamming_threshold}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    # umap_clusters(1, 0, 1000, 32, 0.01)
    # hamming_clusters(1, 0, 1000, 32, 0.01, hamming_eps=2)
    bron_kerbosch_clique_clusters(1, 0, 1000, 32, 0.01, hamming_threshold=10)
    # umap_clusters_twostage(1, 0, 1000, 32, 0.01, cluster_umap_dim=30, viz_umap_dim=2)