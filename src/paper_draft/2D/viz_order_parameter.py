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
    """Extract gamma, alpha, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    if gamma and alpha and mu:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)))
    return (None,)*3

def parse_lattice_line(line):
    """Parse a single line into a 2D lattice array."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    step = int(parts[0])
    lattice_data = parts[1]
    rows = lattice_data.split(';')
    lattice = []
    for row in rows:
        cells = row.split(',')
        lattice_row = []
        for cell in cells:
            bits = tuple(int(b) for b in cell)
            lattice_row.append(bits)
        lattice.append(lattice_row)
    return step, lattice

def compute_largest_cluster_cc3d(lattice_2d):
    """
    Compute largest cluster size using cc3d with periodic boundaries.
    Each unique language gets a unique integer ID for cc3d.
    """
    L = len(lattice_2d)
    
    # Create a mapping from unique languages to integer IDs
    unique_languages = set()
    for i in range(L):
        for j in range(L):
            unique_languages.add(lattice_2d[i][j])
    
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
    
    # Find largest cluster size
    unique_labels = np.unique(labels)
    max_cluster_size = 0
    for label in unique_labels:
        cluster_size = np.sum(labels == label)
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
    
    return max_cluster_size

def process_file_order_parameter(filename):
    """
    Process a single file to compute time-averaged order parameter.
    Order parameter = mean(largest_cluster_size) / (L*L)
    """
    gamma, alpha, mu = extract_params_from_filename(filename)
    
    if gamma is None or alpha is None or mu is None:
        return None
    
    largest_cluster_sizes = []
    L = None
    
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse this single line into a lattice
                step, lattice = parse_lattice_line(line)
                if lattice is None:
                    continue
                
                # Get system size from first lattice
                if L is None:
                    L = len(lattice)
                
                # Compute largest cluster size for this timestep
                largest_cluster_size = compute_largest_cluster_cc3d(lattice)
                largest_cluster_sizes.append(largest_cluster_size)
                
                # Explicitly delete the lattice to free memory
                del lattice
                gc.collect()
        
        # Compute time-averaged order parameter
        if len(largest_cluster_sizes) == 0 or L is None:
            order_parameter = 0.0
        else:
            mean_largest_cluster = np.mean(largest_cluster_sizes)
            order_parameter = mean_largest_cluster / (L * L)
        
        return (gamma, alpha, order_parameter)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_order_parameter(filename)

def load_order_parameter_data_alpha_gamma(L, B, mu):
    """Load the order parameter from each file matching the specified parameters from rasterscan folder."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/rasterscan/L_{L}_B_{B}/g_*_a_*_mu_{mu}.tsv")
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    print(f"Found {len(files)} files in rasterscan/L_{L}_B_{B} folder matching mu={mu}")
    
    # Determine number of processes (leave 4 CPUs free)
    total_cpus = mp.cpu_count()
    num_processes = max(1, total_cpus - 4)
    print(f"Using {num_processes} processes (out of {total_cpus} CPUs)")
    
    results = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for result in pool.imap(worker_process_file, files):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    return results

def create_order_parameter_heatmap(order_data, L, B, mu):
    """Create a heatmap of order parameter organized by gamma (x-axis) and alpha (y-axis)."""
    if not order_data:
        print("No order parameter data to plot")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(order_data, columns=['gamma', 'alpha', 'order_parameter'])
    
    print(f"Successfully processed {len(df)} files")
    print("DataFrame summary:")
    print(df.head())
    
    # Get unique gamma and alpha values
    gammas = np.sort(df['gamma'].unique())
    alphas = np.sort(df['alpha'].unique())
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create grid for heatmap (alpha on y-axis, high to low)
    alphas_reversed = alphas[::-1]  # High to low for y-axis
    order_grid = np.full((len(alphas), len(gammas)), np.nan)
    
    # Fill the grid
    for _, row in df.iterrows():
        gamma_idx = np.where(gammas == row['gamma'])[0][0]
        alpha_idx = np.where(alphas_reversed == row['alpha'])[0][0]
        order_grid[alpha_idx, gamma_idx] = row['order_parameter']
    
    # Create the heatmap with red-blue divergent colormap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(order_grid, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1,
                   extent=[gammas.min()-0.1, gammas.max()+0.1, 
                          alphas.min()-0.1, alphas.max()+0.1])
    
    # Set ticks and labels
    ax.set_xticks(gammas)
    ax.set_yticks(alphas_reversed)
    ax.set_xticklabels([f'{gamma}' for gamma in gammas], fontsize=12)
    ax.set_yticklabels([f'{alpha}' for alpha in alphas_reversed], fontsize=12)
    
    # Customize the plot
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Order Parameter (Largest Cluster Fraction)', rotation=270, labelpad=20, fontsize=14)
    
    ax.set_xlabel('Global Disalignment Strength (γ)', fontsize=16)
    ax.set_ylabel('Local Alignment Strength (α)', fontsize=16)
    ax.set_title(f'Order Parameter vs α and γ (L={L}, B={B}, μ={mu})', fontsize=18)
    
    # Add grid for clarity
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    output_dir = "src/paper_draft/2D/plots/rasterscans"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/order_parameter_heatmap_L_{L}_B_{B}_mu_{mu}.pdf"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fname}")
    
    # Save the DataFrame
    csv_fname = f"{output_dir}/order_parameter_results_L_{L}_B_{B}_mu_{mu}.csv"
    df.to_csv(csv_fname, index=False)
    print(f"Results saved to: {csv_fname}")

def main_order_parameter_alpha_gamma(L=256, B=16, mu=0.0001):
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu} in rasterscan folder")
    order_data = load_order_parameter_data_alpha_gamma(L, B, mu)
    if not order_data:
        print("No valid order parameter data found.")
        return
    print(f"Successfully computed order parameter for {len(order_data)} files")
    create_order_parameter_heatmap(order_data, L, B, mu)

if __name__ == "__main__":
    main_order_parameter_alpha_gamma(L=256, B=16, mu=0.0001)
