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
    """Extract parameter values from filename."""
    # Extract gamma and alpha using regex
    gamma_match = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha_match = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    
    if gamma_match and alpha_match:
        gamma = float(gamma_match.group(1))
        alpha = float(alpha_match.group(1))
        return gamma, alpha
    else:
        return None, None

def load_languages(filename):
    """Load languages from a TSV file and return as binary arrays."""
    try:
        # Read TSV with string data type for language column
        df = pd.read_csv(filename, sep='\t', dtype={'language': str})
        
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
        
        return np.array(languages)
    
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return np.array([])

def compute_clusters(languages, method='umap'):
    """Compute clusters from languages using either UMAP or Hamming distance."""
    if len(languages) < 2:
        return 0, 0  # Not enough languages to cluster
    
    # Get sequence length
    L = languages[0].shape[0]
    
    # Calculate Hamming distance matrix
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L
    
    # Calculate average pairwise Hamming distance for diversity
    avg_hamming = np.sum(hamming_matrix) / (len(languages) * (len(languages) - 1))
    
    if method == 'umap':
        # Calculate diversity ratio (unique languages)
        language_tuples = [tuple(lang) for lang in languages]
        unique_languages = set(language_tuples)
        diversity = len(unique_languages)
        diversity_ratio = diversity / len(languages)
        
        # Run UMAP for dimensionality reduction
        n_neighbors = min(15, len(languages)-1)
        try:
            reducer = umap.UMAP(n_components=2, 
                               n_neighbors=n_neighbors, 
                               min_dist=0.01,
                               metric='precomputed', 
                               random_state=42)
            embedding = reducer.fit_transform(hamming_matrix)
            
            # Calculate pairwise Euclidean distances in the UMAP space
            umap_distances = pairwise_distances(embedding, metric='euclidean')
            
            # Scale epsilon based on diversity - lower diversity means higher epsilon
            base_eps = 0.5
            max_eps = 1.0
            scaled_eps = min(max_eps, base_eps + (1 - diversity_ratio))
            
            # Run DBSCAN on the UMAP embedding
            dbscan = DBSCAN(eps=scaled_eps, min_samples=1, metric='precomputed')
            labels = dbscan.fit_predict(umap_distances)
        except Exception as e:
            print(f"Error in UMAP clustering: {e}")
            return 0, avg_hamming
    
    else:  # hamming method
        # Define what constitutes a meaningful difference
        meaningful_diff_bits = 1
        hamming_eps = meaningful_diff_bits / L
        
        try:
            # Run DBSCAN directly on Hamming distances
            dbscan = DBSCAN(eps=hamming_eps, min_samples=1, metric='precomputed')
            labels = dbscan.fit_predict(hamming_matrix / L)
        except Exception as e:
            print(f"Error in Hamming clustering: {e}")
            return 0, avg_hamming
    
    # Count number of clusters (excluding noise points)
    n_clusters = len(set([l for l in labels if l != -1]))
    
    return n_clusters, avg_hamming

def main():
    # Path to language files
    languages_files_pattern = "src/geneticSimilaritySimplicity/outputs/languages/g_*_a_*_N_*_L_*_mu_*.tsv"
    
    # Get all language files
    language_files = glob.glob(languages_files_pattern)
    
    if not language_files:
        print(f"No files found matching pattern: {languages_files_pattern}")
        return
    
    print(f"Found {len(language_files)} language files")
    
    # Extract parameters and calculate clusters for each file
    results = []
    
    for i, filename in tqdm(enumerate(language_files)):
        print(f"Processing file {i+1}/{len(language_files)}: {os.path.basename(filename)}")
        gamma, alpha = extract_params_from_filename(filename)
        if gamma is not None and alpha is not None:
            languages = load_languages(filename)
            if len(languages) > 0:
                n_clusters, avg_hamming = compute_clusters(languages, method='umap')
                results.append((gamma, alpha, n_clusters, avg_hamming))
                print(f"  Found {n_clusters} clusters, avg Hamming distance: {avg_hamming:.2f}")
    
    # Convert to numpy arrays for easier manipulation
    results = np.array(results)
    
    if len(results) == 0:
        print("No valid results found.")
        return
    
    # Get unique values of gamma and alpha (sorted)
    gammas = np.sort(np.unique(results[:, 0]))
    alphas = np.sort(np.unique(results[:, 1]))
    
    # Create 2D grids for the heatmaps
    clusters_grid = np.zeros((len(gammas), len(alphas)))
    clusters_grid.fill(np.nan)  # Fill with NaN initially
    
    hamming_grid = np.zeros((len(gammas), len(alphas)))
    hamming_grid.fill(np.nan)
    
    # Map each result to the correct position in the grid
    for gamma, alpha, n_clusters, avg_hamming in results:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas == alpha)[0][0]
        clusters_grid[gamma_idx, alpha_idx] = n_clusters
        hamming_grid[gamma_idx, alpha_idx] = avg_hamming
    
    # Create the heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Number of clusters heatmap
    im1 = ax1.imshow(clusters_grid, cmap='viridis', aspect='auto',
                    extent=[min(alphas)-0.5, max(alphas)+0.5, min(gammas)-0.5, max(gammas)+0.5],
                    origin='lower')
    
    # Annotate with actual cluster values
    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            if not np.isnan(clusters_grid[i, j]):
                # Choose text color based on value for readability
                value = clusters_grid[i, j]
                vmin, vmax = np.nanmin(clusters_grid), np.nanmax(clusters_grid)
                if vmin == vmax:
                    brightness = 0.5
                else:
                    brightness = (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                
                ax1.text(alpha, gamma, f"{int(value)}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    
    # Create colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Number of Clusters', rotation=270, labelpad=20)
    
    # Set custom tick positions to match actual parameter values
    ax1.set_xticks(alphas)
    ax1.set_yticks(gammas)
    
    # Add grid lines at tick positions
    ax1.set_xticks(alphas, minor=True)
    ax1.set_yticks(gammas, minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Set labels and title
    ax1.set_xlabel('Alpha (relatedness bonus)')
    ax1.set_ylabel('Gamma (hamming distance penalty)')
    ax1.set_title('Number of Language Clusters Across Parameter Space')
    
    # 2. Average Hamming distance heatmap
    im2 = ax2.imshow(hamming_grid, cmap='plasma', aspect='auto',
                    extent=[min(alphas)-0.5, max(alphas)+0.5, min(gammas)-0.5, max(gammas)+0.5],
                    origin='lower')
    
    # Annotate with actual Hamming distances
    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            if not np.isnan(hamming_grid[i, j]):
                # Choose text color based on value for readability
                value = hamming_grid[i, j]
                vmin, vmax = np.nanmin(hamming_grid), np.nanmax(hamming_grid)
                if vmin == vmax:
                    brightness = 0.5
                else:
                    brightness = (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                
                ax2.text(alpha, gamma, f"{value:.2f}", 
                        ha='center', va='center', 
                        color=text_color, fontweight='bold')
    
    # Create colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance', rotation=270, labelpad=20)
    
    # Set custom tick positions to match actual parameter values
    ax2.set_xticks(alphas)
    ax2.set_yticks(gammas)
    
    # Add grid lines at tick positions
    ax2.set_xticks(alphas, minor=True)
    ax2.set_yticks(gammas, minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Set labels and title
    ax2.set_xlabel('Alpha (relatedness bonus)')
    ax2.set_ylabel('Gamma (hamming distance penalty)')
    ax2.set_title('Average Hamming Distance Across Parameter Space')
    
    # Save the plot
    output_dir = "src/geneticSimilaritySimplicity/plots/languages"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/language_clusters_heatmap.png", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()