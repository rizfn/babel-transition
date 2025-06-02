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

def umap_clusters_understandability_vs_hamming(gamma, alpha, N, L, mu):
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
    base_eps = 0.5
    max_eps = 1.0
    scaled_eps = min(max_eps, base_eps + (1 - diversity_ratio))
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

if __name__ == "__main__":
    umap_clusters_understandability_vs_hamming(3, 1, 1000, 16, 0.01)