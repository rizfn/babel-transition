import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu, K from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9.]+)', filename)
    K = re.search(r'K_([0-9]+)', filename)
    if gamma and alpha and L and B and mu and K:
        return (float(gamma.group(1)), float(alpha.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)), int(K.group(1)))
    return (None,)*6

def parse_lattice_line(line):
    """Parse a single line into a 2D lattice array."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    
    step = int(parts[0])  # First column is the step number
    lattice_data = parts[1]  # Second column contains the lattice
    
    # Parse the lattice data: rows separated by ';', cells by ','
    rows = lattice_data.split(';')
    lattice = []
    for row in rows:
        cells = row.split(',')
        lattice_row = []
        for cell in cells:
            # Each cell is a bitstring, convert to tuple for uniqueness
            bits = tuple(int(b) for b in cell)
            lattice_row.append(bits)
        lattice.append(lattice_row)
    
    # Return as regular list of lists, not numpy array to avoid hashability issues
    return lattice

def get_last_line_from_file(filename):
    """Get the last non-empty line from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Find the last non-empty line
            for line in reversed(lines):
                if line.strip():
                    return line.strip()
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def compute_hamming_distance_distribution_binned(lattice_2d, B, num_bins=15):
    """
    Compute the distribution of Hamming distances as binned histogram data.
    More memory-efficient: bins distances on-the-fly instead of storing all pairs.
    """
    from collections import Counter
    
    L = len(lattice_2d)
    total_agents = L * L
    
    # Flatten the lattice to get all languages
    all_languages = []
    for i in range(L):
        for j in range(L):
            all_languages.append(lattice_2d[i][j])
    
    # Count frequency of each unique bitstring
    bitstring_counts = Counter(all_languages)
    unique_bitstrings = list(bitstring_counts.keys())
    frequencies = list(bitstring_counts.values())
    
    print(f"  Found {len(unique_bitstrings)} unique bitstrings")
    
    # Initialize histogram bins
    bin_edges = np.linspace(0, B, num_bins + 1)
    histogram = np.zeros(num_bins)
    
    # First, handle pairs within the same bitstring group (distance = 0)
    for freq in frequencies:
        if freq > 1:
            # Number of pairs within this group with distance 0
            pairs_within_group = freq * (freq - 1) // 2
            # Distance 0 goes in the first bin
            bin_idx = np.searchsorted(bin_edges[1:], 0)
            if bin_idx < num_bins:
                histogram[bin_idx] += pairs_within_group
    
    if len(unique_bitstrings) < 2:
        # Only one unique bitstring, all distances are 0
        return histogram, bin_edges
    
    # Convert to numpy array for distance computation
    unique_array = np.array(unique_bitstrings)
    
    # Now handle pairs between different bitstring groups
    if len(unique_bitstrings) > 1000:
        print(f"  Large number of unique bitstrings, processing in chunks...")
        chunk_size = 500
        
        for i in range(0, len(unique_bitstrings), chunk_size):
            end_i = min(i + chunk_size, len(unique_bitstrings))
            chunk_i = unique_array[i:end_i]
            freq_i = frequencies[i:end_i]
            
            # Compute distances to all later chunks
            for j in range(end_i, len(unique_bitstrings), chunk_size):
                end_j = min(j + chunk_size, len(unique_bitstrings))
                chunk_j = unique_array[j:end_j]
                freq_j = frequencies[j:end_j]
                
                distances_between = pairwise_distances(chunk_i, chunk_j, metric='hamming') * B
                for ci in range(len(chunk_i)):
                    for cj in range(len(chunk_j)):
                        distance = distances_between[ci, cj]
                        # Weight = number of agent pairs with this distance
                        weight = freq_i[ci] * freq_j[cj]
                        bin_idx = np.searchsorted(bin_edges[1:], distance)
                        if bin_idx < num_bins:
                            histogram[bin_idx] += weight
            
            # Compute distances within this chunk (upper triangle, different bitstrings)
            if len(chunk_i) > 1:
                distances_within = pairwise_distances(chunk_i, metric='hamming') * B
                for ci in range(len(chunk_i)):
                    for cj in range(ci + 1, len(chunk_i)):
                        distance = distances_within[ci, cj]
                        # Weight = number of agent pairs with this distance
                        weight = freq_i[ci] * freq_i[cj]
                        bin_idx = np.searchsorted(bin_edges[1:], distance)
                        if bin_idx < num_bins:
                            histogram[bin_idx] += weight
            
            if i % (chunk_size * 5) == 0:
                print(f"    Processed {i + len(chunk_i)}/{len(unique_bitstrings)} bitstrings")
    
    else:
        # For smaller numbers, compute directly
        hamming_matrix = pairwise_distances(unique_array, metric='hamming') * B
        
        for i in range(len(unique_bitstrings)):
            for j in range(i + 1, len(unique_bitstrings)):
                distance = hamming_matrix[i, j]
                # Weight = number of agent pairs with this distance
                weight = frequencies[i] * frequencies[j]
                bin_idx = np.searchsorted(bin_edges[1:], distance)
                if bin_idx < num_bins:
                    histogram[bin_idx] += weight
        
    return histogram, bin_edges

def process_file_final_timestep(filename):
    """
    Process only the final timestep of a file to get Hamming distance distribution.
    """
    gamma, alpha, L, B, mu, K = extract_params_from_filename(filename)
    
    if gamma is None or alpha is None:
        return None
    
    try:
        print(f"Processing γ={gamma}, α={alpha}...")
        
        # Get only the last line
        last_line = get_last_line_from_file(filename)
        if last_line is None:
            return None
        
        # Parse the final lattice
        lattice = parse_lattice_line(last_line)
        if lattice is None:
            return None
        
        # Compute Hamming distance distribution as binned histogram
        histogram, bin_edges = compute_hamming_distance_distribution_binned(lattice, B, num_bins=15)
        
        # Clean up memory
        del lattice
        gc.collect()
        
        return (gamma, alpha, histogram, bin_edges, L, B, mu, K)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def load_hamming_distributions(L, B, mu, K):
    """Load and compute Hamming distance distributions for all files matching the specified parameters."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscan/L_{L}_g_*_a_*_B_{B}_mu_{mu}_K_{K}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    
    print(f"Found {len(files)} files matching L={L}, B={B}, mu={mu}, K={K}")
    
    # Process files sequentially with progress bar
    results = []
    for filename in tqdm(files, desc="Processing files"):
        result = process_file_final_timestep(filename)
        if result is not None:
            # Validate that the result matches our requested parameters
            gamma, alpha, histogram, bin_edges, L_file, B_file, mu_file, K_file = result
            if L_file == L and B_file == B and mu_file == mu and K_file == K:
                results.append(result)
    
    return results

def create_hamming_histogram_grid(hamming_data, L, B, mu, K):
    """Create a grid of histograms showing Hamming distance distributions organized by gamma and alpha."""
    if not hamming_data:
        print("No Hamming data to plot")
        return
    
    # Organize data by (gamma, alpha)
    data_dict = {}
    
    print("Organizing data...")
    for gamma, alpha, histogram, bin_edges, L_file, B_file, mu_file, K_file in hamming_data:
        data_dict[(gamma, alpha)] = (histogram, bin_edges)
    
    # Get unique gamma and alpha values
    gammas = sorted(set(key[0] for key in data_dict.keys()))
    alphas = sorted(set(key[1] for key in data_dict.keys()), reverse=True)
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    print(f"Grid size: {len(alphas)} x {len(gammas)}")
    
    print("Creating figure...")
    fig, axes = plt.subplots(len(alphas), len(gammas), 
                            figsize=(2*len(gammas), 2*len(alphas)))
    
    # Handle case where we only have one row or column
    if len(alphas) == 1 and len(gammas) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(gammas) == 1:
        axes = [[ax] for ax in axes]
    
    print("Plotting histograms...")
    # Plot each histogram
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            ax = axes[i][j]
            
            if (gamma, alpha) in data_dict:
                histogram, bin_edges = data_dict[(gamma, alpha)]
                
                if np.sum(histogram) > 0:
                    # Plot pre-binned histogram
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.bar(bin_centers, histogram, width=np.diff(bin_edges), 
                           alpha=0.6, color='skyblue', edgecolor='none')
                    
                    # Calculate and display mean
                    if np.sum(histogram) > 0:
                        weighted_mean = np.average(bin_centers, weights=histogram)
                        ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=1)
                        ax.text(0.7, 0.9, f'{weighted_mean:.1f}', 
                               transform=ax.transAxes, fontsize=6, 
                               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No Distances', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=6)
                
                ax.set_title(f'γ={gamma:.1f}, α={alpha:.1f}', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_title(f'γ={gamma:.1f}, α={alpha:.1f}', fontsize=7)
            
            # Set consistent x-axis
            ax.set_xlim(0, B)
            
            # Minimal labels and ticks
            if i == len(alphas) - 1:  # Bottom row
                ax.set_xlabel('Hamming Distance', fontsize=6)
            if j == 0:  # Left column
                ax.set_ylabel('Frequency', fontsize=6)
            
            ax.tick_params(labelsize=5)
    
    # Add overall title
    fig.suptitle(f'Hamming Distance Distributions\n(L={L}, B={B}, μ={mu}, K={K})', fontsize=10)
    
    print("Finalizing plot...")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save the plot
    output_dir = "src/understandabilityVsHamming2D/commutableHamming/plots/hammingDistributions"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/histogram_grid_finalDistributions_L_{L}_B_{B}_mu_{mu}_K_{K}.png"
    
    try:
        plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {fname}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Clean up
    plt.close(fig)
    gc.collect()
    
    print("Plot creation completed")

def main(L=256, B=16, mu=0.001, K=1):
    """Main function that takes parameters and finds matching files."""
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu}, K={K}")
    
    # Load and compute Hamming distance distributions from files matching the specified parameters
    hamming_data = load_hamming_distributions(L, B, mu, K)
    
    if not hamming_data:
        print("No valid Hamming distance data found.")
        return
    
    print(f"Successfully computed Hamming distance distributions for {len(hamming_data)} files")
    
    # Create the histogram grid
    create_hamming_histogram_grid(hamming_data, L, B, mu, K)

if __name__ == "__main__":
    main(L=256, B=16, mu=0.01, K=1)