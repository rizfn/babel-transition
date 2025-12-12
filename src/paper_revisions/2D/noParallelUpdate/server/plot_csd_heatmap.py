import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import sys
import multiprocessing as mp
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract L, B, gamma, alpha, mu from filename and path."""
    # Extract L and B from path
    L_match = re.search(r'CSD_L_(\d+)_B_(\d+)', filename)
    # Extract gamma, alpha, mu from filename
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    
    if L_match and gamma and alpha and mu:
        L = int(L_match.group(1))
        B = int(L_match.group(2))
        return (L, B, float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)))
    return (None, None, None, None, None)

def process_csd_file(filename):
    """
    Process a single CSD file to extract all cluster sizes.
    Format: step\tsize1,size2,size3,...
    """
    L, B, gamma, alpha, mu = extract_params_from_filename(filename)
    
    if L is None or B is None or gamma is None or alpha is None or mu is None:
        return None
    
    try:
        cluster_sizes = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                # Parse cluster sizes (skip step)
                sizes_str = parts[1]
                if sizes_str:
                    sizes = [int(s) for s in sizes_str.split(',')]
                    cluster_sizes.extend(sizes)
        
        return (L, B, gamma, alpha, mu, cluster_sizes)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_csd_file(filename):
    """Wrapper function for multiprocessing."""
    return process_csd_file(filename)

def load_csd_data(B, gamma, mu):
    """Load CSD data from multiple simulation files across different L values."""
    # Pattern to match all L values
    pattern = f"/nbi/nbicmplx/cell/rpw391/babel2D/orderParamL/outputs/CSD_L_*_B_{B}/g_{gamma}_a_*_mu_{mu}_*.tsv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    print(f"Found {len(files)} CSD files matching B={B}, gamma={gamma}, mu={mu}")
    
    # Determine number of processes
    total_cpus = mp.cpu_count()
    num_processes = max(1, total_cpus - 4)
    print(f"Using {num_processes} processes (out of {total_cpus} CPUs)")
    
    results = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(files), desc="Processing CSD files") as pbar:
            for result in pool.imap(worker_process_csd_file, files):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    return results

def aggregate_csd_by_params(csd_data):
    """
    Aggregate cluster sizes for each (L, alpha) combination.
    Returns: dict mapping (L, alpha) -> list of all cluster sizes
    """
    grouped_data = {}
    for L, B, gamma, alpha, mu, cluster_sizes in csd_data:
        key = (L, alpha)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].extend(cluster_sizes)
    
    # Print statistics
    for (L, alpha), sizes in grouped_data.items():
        print(f"L={L}, α={alpha}: {len(sizes)} total cluster observations")
    
    return grouped_data

def plot_csd_grid(aggregated_data, B, gamma, mu, bins=30):
    """
    Create a 2D grid of log-log CSD plots for each (L, alpha) combination.
    L on y-axis (higher = larger L), alpha on x-axis (right = larger alpha).
    """
    if not aggregated_data:
        print("No data to plot")
        return
    
    # Get unique L and alpha values
    params = list(aggregated_data.keys())
    Ls = sorted(set(L for L, a in params), reverse=True)  # High to low for y-axis
    alphas = sorted(set(a for L, a in params))  # Low to high for x-axis
    
    print(f"L values: {Ls}")
    print(f"Alpha values: {alphas}")
    
    n_rows = len(Ls)
    n_cols = len(alphas)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows), facecolor='none')
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.05)
    
    # Create subplots
    for i, L in enumerate(Ls):
        for j, alpha in enumerate(alphas):
            ax = fig.add_subplot(gs[i, j])
            ax.patch.set_facecolor('none')
            
            key = (L, alpha)
            if key in aggregated_data:
                cluster_sizes_array = np.array(aggregated_data[key])
                
                if len(cluster_sizes_array) > 0:
                    min_size = max(1, np.min(cluster_sizes_array))
                    max_size = np.max(cluster_sizes_array)
                    
                    # Log-spaced bins for log-log plot
                    log_bins = np.geomspace(min_size, max_size, bins)
                    log_counts, log_bin_edges = np.histogram(cluster_sizes_array, bins=log_bins)
                    log_bin_centers = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2
                    log_bin_widths = np.diff(log_bin_edges)
                    log_density = log_counts / log_bin_widths  # Normalize by bin width
                    
                    # Filter out zero densities for log-log plot
                    mask = log_density > 0
                    
                    # Plot on log-log scale
                    ax.loglog(log_bin_centers[mask], log_density[mask], 'o-', 
                             markersize=3, linewidth=1, color='#1f77b4', alpha=0.7)
            
            # Set scales
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Remove tick labels except for edges
            if i < n_rows - 1:  # Not bottom row
                ax.set_xticklabels([])
            else:  # Bottom row
                ax.set_xlabel('Cluster Size', fontsize=8)
            
            if j > 0:  # Not left column
                ax.set_yticklabels([])
            else:  # Left column
                ax.set_ylabel('Density', fontsize=8)
            
            # Add parameter labels for edge plots
            if i == 0:  # Top row
                ax.set_title(f'α={alpha:.1f}', fontsize=8, pad=5)
            if j == n_cols - 1:  # Right column
                ax.text(1.05, 0.5, f'L={L}', transform=ax.transAxes,
                       rotation=270, verticalalignment='center', fontsize=8)
            
            # Reduce tick label size
            ax.tick_params(labelsize=6)
            
            # Add grid
            ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    output_dir = f"plots/CSD"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/csd_grid_B_{B}_g_{gamma}_mu_{mu}.svg"
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    print(f"Plot saved to: {fname}")
    plt.close()

def main(B=16, gamma=1.0, mu=0.0001):
    print(f"Processing CSD data for B={B}, gamma={gamma}, mu={mu}")
    
    # Load CSD data
    csd_data = load_csd_data(B, gamma, mu)
    if not csd_data:
        print("No valid CSD data found.")
        return
    
    print(f"Successfully loaded data from {len(csd_data)} CSD files")
    
    # Aggregate by parameters
    aggregated_data = aggregate_csd_by_params(csd_data)
    if not aggregated_data:
        print("No valid aggregated data.")
        return
    
    print(f"Successfully aggregated {len(aggregated_data)} parameter combinations")
    
    # Create the grid plot
    plot_csd_grid(aggregated_data, B, gamma, mu)

if __name__ == "__main__":
    # Default parameters
    B = 16
    gamma = 1
    mu = 0.0001
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        B = int(sys.argv[1])
    if len(sys.argv) > 2:
        gamma = float(sys.argv[2])
    if len(sys.argv) > 3:
        mu = float(sys.argv[3])
    
    main(B, gamma, mu)
