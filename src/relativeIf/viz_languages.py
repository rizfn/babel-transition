import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import matplotlib.colors as mcolors
from sklearn.metrics import pairwise_distances
import os

def hamming(a, b):
    """Calculate the Hamming distance between two binary arrays."""
    return np.sum(a != b)


def umap_clusters_beta(gamma, alpha, max_depth, N, L, mu):
    """Run UMAP first, then clustering with epsilon adjusted by diversity."""
    # Load language data
    base_dir = "src/relativeIf/outputs/beta/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_gdmax_{max_depth}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    # Read TSV with string data type for language column to prevent conversion to numbers
    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Check if the file contains generations column
    if 'generation' in df.columns:
        # Get the highest generation
        last_gen = df['generation'].max()
        # Filter to just the last generation
        df = df[df['generation'] == last_gen]

    # Convert language strings to binary arrays
    languages = []
    for lang_str in df['language']:
        lang_str = str(lang_str)
        lang_array = np.array([int(bit) for bit in lang_str])
        languages.append(lang_array)

    languages = np.array(languages)

    # Calculate diversity (number of unique languages)
    language_tuples = [tuple(lang) for lang in languages]
    unique_languages = set(language_tuples)
    diversity = len(unique_languages)
    diversity_ratio = diversity / len(languages)

    # Create color palette (similar to D3 scheme)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)

    # Use sklearn's pairwise_distances for efficiency
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L

    # Calculate average pairwise Hamming distance
    avg_hamming = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))

    # Run UMAP for dimensionality reduction
    n_neighbors = min(15, len(languages)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01,
                       metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)

    # Calculate pairwise Euclidean distances in the UMAP space
    umap_distances = pairwise_distances(embedding, metric='euclidean')

    # Scale epsilon based on diversity
    base_eps = 0.5
    max_eps = 1.0
    scaled_eps = min(max_eps, base_eps + (1 - diversity_ratio))

    # Run DBSCAN on the UMAP embedding
    dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(umap_distances)

    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Count languages in each cluster
    cluster_counts = {}
    for label in labels:
        if label not in cluster_counts:
            cluster_counts[label] = 0
        cluster_counts[label] += 1

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Create a color mapping dictionary to ensure consistent colors between plots
    color_mapping = {}
    color_mapping[-1] = np.array([0.8, 0.8, 0.8])  # Gray for noise

    # Assign colors to each cluster label
    sorted_clusters = sorted([l for l in set(labels) if l != -1],
                             key=lambda l: -cluster_counts[l])

    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])

    # 1. Plot UMAP with clusters
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                  color=color_mapping[-1], alpha=0.5, s=30)

    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                  color=color_mapping[label], alpha=0.8, s=50)

    gen_info = f"generation {last_gen}" if 'last_gen' in locals() else "final generation"
    ax2.set_title(f'UMAP+DBSCAN Clustering\n'
                 f'{n_clusters} clusters found (γ={gamma}, '
                 f'diversity={diversity_ratio:.2%}, eps={scaled_eps:.2f}, {gen_info})')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')

    # 2. Create colored heatmap based on cluster assignments
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

    output_dir = "src/relativeIf/plots/languages"
    os.makedirs(output_dir, exist_ok=True)

    gen_suffix = f"_gen_{last_gen}" if 'last_gen' in locals() else ""
    output_file = f"{output_dir}/beta_umap_clusters_g_{gamma}_a_{alpha}_gdmax_{max_depth}_N_{N}_L_{L}_mu_{mu}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    umap_clusters_beta(5, 100, 20, 1000, 16, 0.01)