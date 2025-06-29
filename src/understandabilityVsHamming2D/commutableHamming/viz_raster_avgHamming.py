import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
import multiprocessing as mp
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

def compute_average_hamming_distance(lattice_2d, B):
    """
    Compute the average Hamming distance between all pairs of languages in the lattice.
    Uses frequency-weighted unique bitstrings for efficiency.
    """
    from collections import Counter
    
    L = len(lattice_2d)
    
    # Flatten the lattice to get all languages
    all_languages = []
    for i in range(L):
        for j in range(L):
            all_languages.append(lattice_2d[i][j])
    
    # Count frequency of each unique bitstring
    bitstring_counts = Counter(all_languages)
    unique_bitstrings = list(bitstring_counts.keys())
    frequencies = list(bitstring_counts.values())
    
    if len(unique_bitstrings) < 2:
        return 0.0
    
    # Convert to numpy array for distance computation
    unique_array = np.array(unique_bitstrings)
    
    # Compute pairwise Hamming distances between unique bitstrings
    hamming_matrix = pairwise_distances(unique_array, metric='hamming') * B
    
    # Weight distances by frequency pairs
    total_weighted_distance = 0.0
    total_pairs = 0
    
    for i in range(len(unique_bitstrings)):
        for j in range(len(unique_bitstrings)):
            if i != j:  # Exclude diagonal (same bitstring pairs)
                distance = hamming_matrix[i, j]
                weight = frequencies[i] * frequencies[j]
                total_weighted_distance += distance * weight
                total_pairs += weight
    
    if total_pairs == 0:
        return 0.0
    
    avg_hamming = total_weighted_distance / total_pairs
    return avg_hamming

def process_file_memory_efficient(filename):
    """
    Process a single file line-by-line to compute average Hamming distance.
    Memory efficient: processes one line at a time and accumulates statistics.
    """
    gamma, alpha, L, B, mu, K = extract_params_from_filename(filename)
    
    if gamma is None or alpha is None:
        return None
    
    # Accumulate statistics for all timesteps
    total_hamming_sum = 0.0
    timestep_count = 0
    
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse this single line into a lattice
                lattice = parse_lattice_line(line)
                if lattice is None:
                    continue
                
                # Compute average Hamming distance for this timestep
                avg_hamming = compute_average_hamming_distance(lattice, B)
                
                total_hamming_sum += avg_hamming
                timestep_count += 1
                
                # Explicitly delete the lattice to free memory
                del lattice
                gc.collect()
        
        # Compute average Hamming distance across all timesteps
        if timestep_count == 0:
            avg_hamming_distance = 0.0
        else:
            avg_hamming_distance = total_hamming_sum / timestep_count
        
        return (gamma, alpha, avg_hamming_distance, L, B, mu, K)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_memory_efficient(filename)

def load_hamming_data(L, B, mu, K):
    """Load and compute Hamming distance data for all files matching the specified parameters."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscan/L_{L}_g_*_a_*_B_{B}_mu_{mu}_K_{K}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    
    print(f"Found {len(files)} files matching L={L}, B={B}, mu={mu}, K={K}")
    
    # Determine number of processes (leave 4 CPUs free)
    total_cpus = mp.cpu_count()
    num_processes = max(1, total_cpus - 4)
    print(f"Using {num_processes} processes (out of {total_cpus} CPUs)")
    
    # Process files in parallel
    results = []
    with mp.Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for result in pool.imap(worker_process_file, files):
                if result is not None:
                    # Validate that the result matches our requested parameters
                    gamma, alpha, avg_hamming_distance, L_file, B_file, mu_file, K_file = result
                    if L_file == L and B_file == B and mu_file == mu and K_file == K:
                        results.append(result)
                pbar.update(1)
    
    return results

def create_hamming_heatmap(hamming_data, L, B, mu, K):
    """Create a heatmap of average Hamming distances organized by gamma and alpha."""
    if not hamming_data:
        print("No Hamming data to plot")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(hamming_data, columns=['gamma', 'alpha', 'avg_hamming_distance', 'L', 'B', 'mu', 'K'])
    
    print(f"Successfully processed {len(df)} files")
    print("DataFrame summary:")
    print(df.head())
    
    # Get unique gamma and alpha values
    gammas = np.sort(df['gamma'].unique())
    alphas = np.sort(df['alpha'].unique())
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create grid for heatmap
    hamming_grid = np.full((len(alphas), len(gammas)), np.nan)
    
    # Fill the grid
    for _, row in df.iterrows():
        gamma_idx = np.where(gammas == row['gamma'])[0][0]
        alpha_idx = np.where(alphas == row['alpha'])[0][0]
        hamming_grid[alpha_idx, gamma_idx] = row['avg_hamming_distance']
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(hamming_grid, cmap='plasma', aspect='auto',
                   extent=[min(gammas)-0.5, max(gammas)+0.5, 
                          min(alphas)-0.5, max(alphas)+0.5],
                   origin='lower')
    
    # Add text annotations
    x_centers = np.linspace(min(gammas), max(gammas), len(gammas))
    y_centers = np.linspace(min(alphas), max(alphas), len(alphas))
    
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(hamming_grid[i, j]):
                value = hamming_grid[i, j]
                # Determine text color based on background brightness
                vmin, vmax = np.nanmin(hamming_grid), np.nanmax(hamming_grid)
                if vmin == vmax:
                    brightness = 0.5
                else:
                    brightness = (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                
                ax.text(x_centers[j], y_centers[i], f"{value:.2f}", 
                       ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    # Customize the plot
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Hamming Distance', rotation=270, labelpad=20)
    
    ax.set_xticks(gammas)
    ax.set_yticks(alphas)
    ax.set_xticks(gammas, minor=True)
    ax.set_yticks(alphas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Gamma (global interaction strength)')
    ax.set_ylabel('Alpha (local interaction strength)')
    ax.set_title(f'Average Hamming Distance vs Alpha and Gamma\n(L={L}, B={B}, μ={mu}, K={K})')
    
    # Save the plot with parameters in filename
    output_dir = "src/understandabilityVsHamming2D/commutableHamming/plots/hammingDistance"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/heatmap_avgHammingDistance_L_{L}_B_{B}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")
    
    # Save the DataFrame for future use with parameters in filename
    csv_fname = f"{output_dir}/avg_hamming_distance_results_L_{L}_B_{B}_mu_{mu}_K_{K}.csv"
    df.to_csv(csv_fname, index=False)
    print(f"Results saved to: {csv_fname}")

def main(L=256, B=16, mu=0.001, K=1):
    """Main function that takes parameters and finds matching files."""
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu}, K={K}")
    
    # Load and compute Hamming distance data from files matching the specified parameters
    hamming_data = load_hamming_data(L, B, mu, K)
    
    if not hamming_data:
        print("No valid Hamming distance data found.")
        return
    
    print(f"Successfully computed Hamming distance data for {len(hamming_data)} files")
    
    # Create the heatmap
    create_hamming_heatmap(hamming_data, L, B, mu, K)

if __name__ == "__main__":
    main(L=256, B=16, mu=0.0001, K=1)
    