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


def hamming_clusters(gamma, N, L, mu):
    # Load language data
    base_dir = "src/geneticSimilarity/outputs/languages"
    languages_file = f"{base_dir}/g_{gamma}_N_{N}_L_{L}_mu_{mu}.tsv"
    
    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return
    
    print(f"Loading languages from {languages_file}")
    
    # Read TSV with string data type for language column to prevent conversion to numbers
    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})
    
    # Check if the file contains generations column
    if 'generation' in df.columns:
        # Get the highest generation
        last_gen = df['generation'].max()
        print(f"Multiple generations found. Selecting only the last generation: {last_gen}")
        # Filter to just the last generation
        df = df[df['generation'] == last_gen]
    else:
        print("No generation column found. Assuming single generation data.")
    
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
    
    # Calculate average pairwise Hamming distance
    diversity = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))
    print(f"Average Hamming distance: {diversity:.2f} bits out of {L}")
    
    # Run DBSCAN directly on the Hamming distances with absolute threshold
    print("Clustering with DBSCAN in Hamming space...")
    # Define what constitutes a meaningful difference: languages that differ by <= meaningful_diff_bits are considered similar
    meaningful_diff_bits = 1
    hamming_eps = meaningful_diff_bits / L  # Convert to normalized hamming distance [0-1]
    dbscan_hamming = DBSCAN(eps=hamming_eps, min_samples=1, metric='precomputed')
    labels = dbscan_hamming.fit_predict(hamming_matrix / L)  # Normalize to [0-1] range
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters using Hamming distance threshold of {meaningful_diff_bits} bits")
    
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
    
    # Create title with parameter values
    gen_info = f"generation {last_gen}" if 'last_gen' in locals() else "final generation"
    ax2.set_title(f'UMAP Visualization with Hamming-based Clustering\n'
                 f'{n_clusters} clusters found (γ={gamma}, threshold={meaningful_diff_bits} bits, {gen_info})')
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
    
    # Create output directory if it doesn't exist
    output_dir = "src/geneticSimilarity/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get generation info for filename
    gen_suffix = f"_gen_{last_gen}" if 'last_gen' in locals() else ""
    
    output_file = f"{output_dir}/hamming_clusters_g_{gamma}_N_{N}_L_{L}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()


def umap_clusters(gamma, N, L, mu):
    """Run UMAP first, then clustering with epsilon adjusted by diversity."""
    # Load language data
    base_dir = "src/geneticSimilarity/outputs/languages"
    languages_file = f"{base_dir}/g_{gamma}_N_{N}_L_{L}_mu_{mu}.tsv"
    
    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return
    
    print(f"Loading languages from {languages_file}")
    
    # Read TSV with string data type for language column to prevent conversion to numbers
    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})
    
    # Check if the file contains generations column
    if 'generation' in df.columns:
        # Get the highest generation
        last_gen = df['generation'].max()
        print(f"Multiple generations found. Selecting only the last generation: {last_gen}")
        # Filter to just the last generation
        df = df[df['generation'] == last_gen]
    else:
        print("No generation column found. Assuming single generation data.")
    
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
    
    # Calculate diversity (number of unique languages)
    # Convert numpy arrays to tuples for hashability
    language_tuples = [tuple(lang) for lang in languages]
    unique_languages = set(language_tuples)
    diversity = len(unique_languages)
    diversity_ratio = diversity / len(languages)
    print(f"Diversity: {diversity} unique languages out of {len(languages)} ({diversity_ratio:.2%})")
    
    # Create color palette (similar to D3 scheme)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    
    # Calculate Hamming distance matrix
    print("Calculating Hamming distances...")
    # Use sklearn's pairwise_distances for efficiency
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    
    # Calculate average pairwise Hamming distance
    avg_hamming = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))
    print(f"Average Hamming distance: {avg_hamming:.2f} bits out of {L}")
    
    # Run UMAP for dimensionality reduction
    print("Running UMAP for dimensionality reduction...")
    n_neighbors = min(15, len(languages)-1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.01, 
                       metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(hamming_matrix)
    
    # Now cluster in the UMAP space
    print("Clustering in UMAP space...")
    
    # Calculate pairwise Euclidean distances in the UMAP space
    umap_distances = pairwise_distances(embedding, metric='euclidean')
    
    # Scale epsilon based on diversity - lower diversity means we should have a HIGHER epsilon
    # Base epsilon is 0.5, inversely scale it by diversity ratio with a maximum of 1.0
    base_eps = 0.5
    max_eps = 1.0
    
    # If diversity is low (ratio close to 0), epsilon should be high
    # If diversity is high (ratio close to 1), epsilon should be lower
    scaled_eps = min(max_eps, base_eps + (1 - diversity_ratio))
    
    print(f"Using scaled epsilon: {scaled_eps:.3f} (diversity ratio: {diversity_ratio:.2f})")
    
    # Run DBSCAN on the UMAP embedding
    dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(umap_distances)
    
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters using UMAP+DBSCAN")
    
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
    
    # Create title with parameter values
    gen_info = f"generation {last_gen}" if 'last_gen' in locals() else "final generation"
    ax2.set_title(f'UMAP+DBSCAN Clustering\n'
                 f'{n_clusters} clusters found (γ={gamma}, ' 
                 f'diversity={diversity_ratio:.2%}, eps={scaled_eps:.2f}, {gen_info})')
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
    
    # Create output directory if it doesn't exist
    output_dir = "src/geneticSimilarity/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get generation info for filename
    gen_suffix = f"_gen_{last_gen}" if 'last_gen' in locals() else ""
    
    output_file = f"{output_dir}/umap_clusters_g_{gamma}_N_{N}_L_{L}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()


def umap_clusters_beta(gamma, N, L, mu):
    """Run UMAP first, then clustering with epsilon adjusted by diversity."""
    # Load language data
    base_dir = "src/geneticSimilarity/outputs/beta/languages"
    languages_file = f"{base_dir}/g_{gamma}_N_{N}_L_{L}_mu_{mu}.tsv"
    
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
        # Make sure we have a string
        lang_str = str(lang_str)
        # Convert each character to an integer
        lang_array = np.array([int(bit) for bit in lang_str])
        languages.append(lang_array)
    
    languages = np.array(languages)
    
    # Calculate diversity (number of unique languages)
    # Convert numpy arrays to tuples for hashability
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
    
    # Scale epsilon based on diversity - lower diversity means we should have a HIGHER epsilon
    # Base epsilon is 0.5, inversely scale it by diversity ratio with a maximum of 1.0
    base_eps = 0.5
    max_eps = 1.0
    
    # If diversity is low (ratio close to 0), epsilon should be high
    # If diversity is high (ratio close to 1), epsilon should be lower
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
    
    # Create title with parameter values
    gen_info = f"generation {last_gen}" if 'last_gen' in locals() else "final generation"
    ax2.set_title(f'UMAP+DBSCAN Clustering\n'
                 f'{n_clusters} clusters found (γ={gamma}, ' 
                 f'diversity={diversity_ratio:.2%}, eps={scaled_eps:.2f}, {gen_info})')
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
    
    # Create output directory if it doesn't exist
    output_dir = "src/geneticSimilarity/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get generation info for filename
    gen_suffix = f"_gen_{last_gen}" if 'last_gen' in locals() else ""
    
    output_file = f"{output_dir}/beta_umap_clusters_g_{gamma}_N_{N}_L_{L}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    # hamming_clusters(-4, 1000, 16, 0.01)
    # umap_clusters(-0.05, 1000, 16, 0.01)
    umap_clusters_beta(0, 1000, 16, 0.01)
