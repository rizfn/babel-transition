import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
import multiprocessing as mp
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, mu, timestamp from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    timestamp = re.search(r'_(\d+)\.tsv$', filename)
    if gamma and alpha and mu and timestamp:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)), int(timestamp.group(1)))
    return (None,)*4

def process_order_param_file(filename):
    """
    Process a single order parameter file to compute time-averaged order parameter.
    New format: step\tlargest_cluster_fraction
    """
    gamma, alpha, mu, timestamp = extract_params_from_filename(filename)
    
    if gamma is None or alpha is None or mu is None or timestamp is None:
        return None
    
    try:
        # Read the file, skipping header if present
        df = pd.read_csv(filename, sep='\t')
        
        # Handle both with and without headers
        if 'largest_cluster_fraction' in df.columns:
            order_parameter_values = df['largest_cluster_fraction'].values
        else:
            # Assume second column is the order parameter
            order_parameter_values = df.iloc[:, 1].values
        
        # Compute time-averaged order parameter for this file
        mean_order_parameter = np.mean(order_parameter_values)
        
        return (gamma, alpha, mean_order_parameter, timestamp)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_order_param_file(filename)

def load_largest_cluster_data_alpha_gamma(L, B, mu):
    """Load order parameter data from multiple simulation files for each parameter combination."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/rasterscanLargestCluster/L_{L}_B_{B}/g_*_a_*_mu_{mu}_*.tsv")
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    print(f"Found {len(files)} files in rasterscanLargestCluster/L_{L}_B_{B} folder matching mu={mu}")
    
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

def aggregate_simulations(simulation_data):
    """
    Aggregate multiple simulations for each (gamma, alpha) combination.
    Returns: List of (gamma, alpha, mean_order_param, std_error)
    """
    # Group by (gamma, alpha)
    grouped_data = {}
    for gamma, alpha, order_param, timestamp in simulation_data:
        key = (gamma, alpha)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(order_param)
    
    # Calculate mean and standard error for each group
    aggregated_results = []
    for (gamma, alpha), order_params in grouped_data.items():
        order_params = np.array(order_params)
        mean_order_param = np.mean(order_params)
        std_error = np.std(order_params, ddof=1) / np.sqrt(len(order_params)) if len(order_params) > 1 else 0
        aggregated_results.append((gamma, alpha, mean_order_param, std_error))
        print(f"γ={gamma}, α={alpha}: {len(order_params)} sims, mean={mean_order_param:.4f}, SE={std_error:.4f}")
    
    return aggregated_results

def create_order_parameter_heatmap(aggregated_data, L, B, mu):
    """Create a heatmap of order parameter organized by gamma (x-axis) and alpha (y-axis)."""
    if not aggregated_data:
        print("No order parameter data to plot")
        return
    
    # Convert to arrays for easier handling
    gammas_all = np.array([item[0] for item in aggregated_data])
    alphas_all = np.array([item[1] for item in aggregated_data])
    order_params = np.array([item[2] for item in aggregated_data])
    std_errors = np.array([item[3] for item in aggregated_data])
    
    print(f"Successfully processed {len(aggregated_data)} parameter combinations")
    
    # Get unique gamma and alpha values
    gammas = np.sort(np.unique(gammas_all))
    alphas = np.sort(np.unique(alphas_all))
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create grid for heatmap (alpha on y-axis, high to low)
    alphas_reversed = alphas[::-1]  # High to low for y-axis
    order_grid = np.full((len(alphas), len(gammas)), np.nan)
    error_grid = np.full((len(alphas), len(gammas)), np.nan)
    
    # Fill the grids
    for gamma, alpha, order_param, std_error in aggregated_data:
        gamma_idx = np.where(gammas == gamma)[0][0]
        alpha_idx = np.where(alphas_reversed == alpha)[0][0]
        order_grid[alpha_idx, gamma_idx] = order_param
        error_grid[alpha_idx, gamma_idx] = std_error
    
    # Create the heatmap with red-blue divergent colormap and transparent face color
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='none')
    ax.patch.set_facecolor('none')
    
    im = ax.imshow(order_grid, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1,
                   extent=[gammas.min()-0.1, gammas.max()+0.1, 
                          alphas.min()-0.1, alphas.max()+0.1])
    
    # Set ticks and labels - only show min and max
    ax.set_xticks([gammas.min(), gammas.max()])
    ax.set_yticks([alphas.min(), alphas.max()])
    ax.set_xticklabels([f'{gammas.min()}', f'{gammas.max()}'], fontsize=30)
    ax.set_yticklabels([f'{alphas.min()}', f'{alphas.max()}'], fontsize=30)
    
    # Customize the plot with closer labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Largest Cluster Fraction', rotation=270, labelpad=0, fontsize=30)
    # Set colorbar ticks to only show min and max
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0.0', '1.0'], fontsize=30)
    
    ax.set_xlabel('γ', fontsize=40, labelpad=-25)
    ax.set_ylabel('α', fontsize=40, labelpad=-35)
    
    # Add grid for clarity
    ax.grid(True, alpha=0.3)
    
    # Save the plot with transparent background
    output_dir = "src/paper_draft/2D/plots/rasterscans"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/largest_cluster_heatmap_L_{L}_B_{B}_mu_{mu}.svg"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    print(f"Plot saved to: {fname}")

def main_largest_cluster_alpha_gamma(L=256, B=16, mu=0.0001):
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu} in rasterscanLargestCluster folder")
    
    # Load simulation data
    simulation_data = load_largest_cluster_data_alpha_gamma(L, B, mu)
    if not simulation_data:
        print("No valid largest cluster data found.")
        return
    
    print(f"Successfully loaded data from {len(simulation_data)} simulation files")
    
    # Aggregate multiple simulations for each parameter combination
    aggregated_data = aggregate_simulations(simulation_data)
    if not aggregated_data:
        print("No valid aggregated data.")
        return
    
    print(f"Successfully aggregated {len(aggregated_data)} parameter combinations")
    
    # Create the heatmap
    create_order_parameter_heatmap(aggregated_data, L, B, mu)

if __name__ == "__main__":
    main_largest_cluster_alpha_gamma(L=256, B=16, mu=0.0001)
