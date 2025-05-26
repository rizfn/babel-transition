import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import networkx as nx
from sklearn.metrics import pairwise_distances
import matplotlib.colors as mcolors

# Parameters for clustering
MEANINGFUL_DIFF_BITS = 8  # Hamming distance threshold for clustering

def hamming(a, b):
    """Calculate the Hamming distance between two binary arrays."""
    return np.sum(a != b)

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
    
    # Create a graph where edges connect points within threshold distance
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] <= threshold:
                G.add_edge(i, j)
    
    # Find all maximal cliques (fully connected subgraphs)
    # This ensures all points in a cluster are within threshold of each other
    cliques = list(nx.find_cliques(G))
        
    # Sort cliques by size (largest first)
    cliques.sort(key=len, reverse=True)
    
    # Assign cluster labels
    labels = np.full(n, -1)  # Default to noise
    
    for i, clique in enumerate(cliques):
        # Only assign points that haven't been assigned yet
        # This ensures a point is only in one cluster
        for point in clique:
            if labels[point] == -1:
                labels[point] = i
    
    return labels

def main():
    # Parameters
    gamma = -10
    N = 1000
    L = 16
    mu = 0.01
    
    # Load language data
    languages_file = f"src/evolveHamming/outputs/languages_g_{gamma}_N_{N}_L_{L}_mu_{mu}.tsv"
    print(f"Loading languages from {languages_file}")
    
    # Read TSV with string data type for language column to prevent conversion to numbers
    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})
        
    # Convert language strings to binary arrays
    languages = []
    for lang_str in df['language']:
        # Make sure we have a string
        lang_str = str(lang_str)
        # Convert each character to an integer
        lang_array = np.array([int(bit) for bit in lang_str])
        languages.append(lang_array)
    
    languages = np.array(languages)
    print(f"Loaded {len(languages)} languages, each with {languages[0].shape[0]} bits")
    
    # Create color palette (similar to D3 scheme)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    
    # Calculate Hamming distance matrix
    print("Calculating Hamming distances...")
    # Use sklearn's pairwise_distances for efficiency
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    
    # Run clique-based clustering with absolute threshold
    print("Running clique-based clustering in Hamming space...")
    labels = find_clique_clusters(hamming_matrix, MEANINGFUL_DIFF_BITS)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clique clusters using Hamming distance threshold of {MEANINGFUL_DIFF_BITS} bits")
    
    # Count languages in each cluster
    cluster_counts = {}
    for label in labels:
        if label not in cluster_counts:
            cluster_counts[label] = 0
        cluster_counts[label] += 1
    
    print("\nCluster statistics:")
    for label, count in sorted(cluster_counts.items()):
        if label == -1:
            print(f"  Noise points: {count} ({count/len(labels):.1%})")
        else:
            print(f"  Cluster {label}: {count} languages ({count/len(labels):.1%})")
    
    # Now run UMAP for visualization only (not for clustering)
    print("Running UMAP for visualization...")
    n_neighbors = min(15, len(languages)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, 
                      metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create a color mapping dictionary to ensure consistent colors between plots
    color_mapping = {}
    color_mapping[-1] = np.array([0.8, 0.8, 0.8])  # Gray for noise
    
    # Assign colors to each cluster label
    # Sort clusters by size for consistent coloring (largest cluster gets first color)
    sorted_clusters = sorted([l for l in set(labels) if l != -1], 
                             key=lambda l: -cluster_counts[l])
    
    for i, label in enumerate(sorted_clusters):
        color_mapping[label] = np.array(colors[i % len(colors)])
    
    # 1. Plot UMAP with clusters
    # Plot noise points first
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax2.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                  color=color_mapping[-1], alpha=0.5, s=30)
    
    # Then plot clusters using our color mapping
    for label in sorted_clusters:
        mask = labels == label
        ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                  color=color_mapping[label], alpha=0.8, s=50)
    
    ax2.set_title(f'UMAP Visualization with Clique-based Clustering\n{n_clusters} clusters found (γ={gamma}, threshold={MEANINGFUL_DIFF_BITS} bits)')
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')
    
    # 2. Create colored heatmap based on cluster assignments
    # Sort languages by cluster for better visualization
    sorted_indices = np.argsort(labels)
    sorted_languages = languages[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Create a colored language matrix where each bit position is colored by its cluster
    # Initialize with white (1,1,1) instead of black (0,0,0)
    language_colors = np.ones((len(sorted_languages), L, 3))
    
    # Assign colors based on cluster labels using the same color mapping
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]  # Get the same color used in the scatter plot
        
        # For each bit that is 1, apply the color
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    
    ax1.imshow(language_colors, aspect='auto', interpolation='none',
               extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    
    ax1.set_title(f'Languages Heatmap (All {len(languages)} Languages)')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Language ID (sorted by cluster)')
    
    plt.tight_layout()
    # Update filename to reflect clique-based clustering
    output_file = f"src/evolveHamming/plots/languages/clique_clusters_g_{gamma}_N_{N}_L_{L}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()