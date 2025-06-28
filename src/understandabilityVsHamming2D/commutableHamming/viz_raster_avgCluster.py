import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cc3d
import re
import gc
import multiprocessing as mp
from tqdm import tqdm

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

def compute_clusters_cc3d(lattice_2d):
    """
    Compute clusters using cc3d with periodic boundaries.
    Each unique language gets a unique integer ID for cc3d.
    """
    L = len(lattice_2d)  # Use len() instead of .shape since it's now a list
    
    # Create a mapping from unique languages to integer IDs
    unique_languages = set()
    for i in range(L):
        for j in range(L):
            unique_languages.add(lattice_2d[i][j])  # Now accessing as list[i][j]
    
    # Convert set to sorted list for consistent mapping
    unique_languages = sorted(list(unique_languages))
    # Add 1 to all IDs so they start from 1 (cc3d treats 0 as background)
    lang_to_id = {lang: idx + 1 for idx, lang in enumerate(unique_languages)}
    
    # Create integer lattice for cc3d
    int_lattice = np.zeros((L, L), dtype=np.int32)
    for i in range(L):
        for j in range(L):
            int_lattice[i, j] = lang_to_id[lattice_2d[i][j]]
    
    labels = cc3d.connected_components(int_lattice, connectivity=4, periodic_boundary=True, return_N=False)
    
    # Count unique cluster labels (no need to exclude 0 anymore since we start from 1)
    unique_labels = np.unique(labels)
    cluster_sizes = []
    for label in unique_labels:
        cluster_size = np.sum(labels == label)
        cluster_sizes.append(cluster_size)
    
    return cluster_sizes

def compute_sqrt_weighted_cluster_size(cluster_sizes):
    """
    Compute the square root of the expectation value of cluster sizes squared: √(E[A²]) = √(Σ(Aᵢ²)/Σ(Aᵢ))
    where Aᵢ is the size of cluster i.
    """
    if len(cluster_sizes) == 0:
        return 0.0
    
    cluster_sizes = np.array(cluster_sizes)
    total_area = np.sum(cluster_sizes)
    weighted_sum = np.sum(cluster_sizes**2)
    
    if total_area == 0:
        return 0.0
    
    expectation_squared = weighted_sum / total_area
    return np.sqrt(expectation_squared)

def process_file_memory_efficient(filename):
    """
    Process a single file line-by-line to compute sqrt of weighted average cluster size.
    Memory efficient: processes one line at a time and accumulates cluster size statistics.
    """
    gamma, alpha, L, B, mu, K = extract_params_from_filename(filename)
    
    if gamma is None or alpha is None:
        return None
    
    # Accumulate statistics for all timesteps
    total_area_sum = 0
    total_weighted_sum = 0
    
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
                
                # Compute cluster sizes for this timestep
                cluster_sizes = compute_clusters_cc3d(lattice)
                
                if len(cluster_sizes) > 0:
                    cluster_sizes = np.array(cluster_sizes)
                    total_area_sum += np.sum(cluster_sizes)
                    total_weighted_sum += np.sum(cluster_sizes**2)
                
                # Explicitly delete the lattice to free memory
                del lattice
                gc.collect()
        
        # Compute sqrt of weighted average cluster size across all timesteps
        if total_area_sum == 0:
            sqrt_weighted_cluster_size = 0.0
        else:
            expectation_squared = total_weighted_sum / total_area_sum
            sqrt_weighted_cluster_size = np.sqrt(expectation_squared)
        
        return (gamma, alpha, sqrt_weighted_cluster_size, L, B, mu, K)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_memory_efficient(filename)

def load_cluster_data(L, B, mu, K):
    """Load and compute cluster data for all files matching the specified parameters."""
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
                    gamma, alpha, sqrt_weighted_cluster_size, L_file, B_file, mu_file, K_file = result
                    if L_file == L and B_file == B and mu_file == mu and K_file == K:
                        results.append(result)
                pbar.update(1)
    
    return results

def create_cluster_heatmap(cluster_data, L, B, mu, K):
    """Create a heatmap of sqrt weighted cluster sizes organized by gamma and alpha."""
    if not cluster_data:
        print("No cluster data to plot")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(cluster_data, columns=['gamma', 'alpha', 'sqrt_weighted_cluster_size', 'L', 'B', 'mu', 'K'])
    
    print(f"Successfully processed {len(df)} files")
    print("DataFrame summary:")
    print(df.head())
    
    # Get unique gamma and alpha values
    gammas = np.sort(df['gamma'].unique())
    alphas = np.sort(df['alpha'].unique())
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create grid for heatmap
    cluster_grid = np.full((len(alphas), len(gammas)), np.nan)
    
    # Fill the grid
    for _, row in df.iterrows():
        gamma_idx = np.where(gammas == row['gamma'])[0][0]
        alpha_idx = np.where(alphas == row['alpha'])[0][0]
        cluster_grid[alpha_idx, gamma_idx] = row['sqrt_weighted_cluster_size']
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(cluster_grid, cmap='viridis', aspect='auto',
                   extent=[min(gammas)-0.5, max(gammas)+0.5, 
                          min(alphas)-0.5, max(alphas)+0.5],
                   origin='lower')
    
    # Add text annotations
    x_centers = np.linspace(min(gammas), max(gammas), len(gammas))
    y_centers = np.linspace(min(alphas), max(alphas), len(alphas))
    
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(cluster_grid[i, j]):
                value = cluster_grid[i, j]
                # Determine text color based on background brightness
                vmin, vmax = np.nanmin(cluster_grid), np.nanmax(cluster_grid)
                if vmin == vmax:
                    brightness = 0.5
                else:
                    brightness = (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                
                ax.text(x_centers[j], y_centers[i], f"{value:.1f}", 
                       ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    # Customize the plot
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('√(E[Cluster Size²])', rotation=270, labelpad=20)
    
    ax.set_xticks(gammas)
    ax.set_yticks(alphas)
    ax.set_xticks(gammas, minor=True)
    ax.set_yticks(alphas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Gamma (global interaction strength)')
    ax.set_ylabel('Alpha (local interaction strength)')
    ax.set_title(f'Square Root of Weighted Cluster Size vs Alpha and Gamma\n(L={L}, B={B}, μ={mu}, K={K})')
    
    # Save the plot with parameters in filename
    output_dir = "src/understandabilityVsHamming2D/commutableHamming/plots/clusterSizes"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/heatmap_sqrtWeightedClusterSize_L_{L}_B_{B}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")
    
    # Save the DataFrame for future use with parameters in filename
    csv_fname = f"{output_dir}/sqrt_weighted_cluster_size_results_L_{L}_B_{B}_mu_{mu}_K_{K}.csv"
    df.to_csv(csv_fname, index=False)
    print(f"Results saved to: {csv_fname}")

def main(L=256, B=16, mu=0.001, K=1):
    """Main function that takes parameters and finds matching files."""
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu}, K={K}")
    
    # Load and compute cluster data from files matching the specified parameters
    cluster_data = load_cluster_data(L, B, mu, K)
    
    if not cluster_data:
        print("No valid cluster data found.")
        return
    
    print(f"Successfully computed cluster data for {len(cluster_data)} files")
    
    # Create the heatmap
    create_cluster_heatmap(cluster_data, L, B, mu, K)

if __name__ == "__main__":
    main(L=256, B=16, mu=0.1, K=1)
