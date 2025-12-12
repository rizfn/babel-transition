import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
import sys
import multiprocessing as mp
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract L, B, gamma, alpha, mu, timestamp from filename and path."""
    # Extract L and B from path
    L_match = re.search(r'/L_(\d+)_B_(\d+)/', filename)
    # Extract gamma, alpha, mu, timestamp from filename
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    timestamp = re.search(r'_(\d+)\.tsv$', filename)
    
    if L_match and gamma and alpha and mu and timestamp:
        L = int(L_match.group(1))
        B = int(L_match.group(2))
        return (L, B, float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)), int(timestamp.group(1)))
    return (None, None, None, None, None, None)

def process_cluster_analysis_file(filename):
    """
    Process a single cluster analysis file to compute time-averaged metrics.
    Format: step\tnumber_of_clusters\tlargest_cluster_fraction
    """
    L, B, gamma, alpha, mu, timestamp = extract_params_from_filename(filename)

    if L is None or B is None or gamma is None or alpha is None or mu is None or timestamp is None:
        return None

    try:
        # Read the file, skipping header if present
        df = pd.read_csv(filename, sep='\t')

        # Handle both with and without headers
        if ('number_of_clusters' in df.columns and
            'largest_cluster_fraction' in df.columns):
            num_clusters_values = df['number_of_clusters'].values
            largest_cluster_values = df['largest_cluster_fraction'].values
        else:
            # Assume second and third columns are the metrics
            num_clusters_values = df.iloc[:, 1].values
            largest_cluster_values = df.iloc[:, 2].values

        # Compute time-averaged metrics for this file
        mean_num_clusters = np.mean(num_clusters_values)
        mean_largest_cluster = np.mean(largest_cluster_values)

        return (L, B, gamma, alpha, mu, mean_num_clusters, mean_largest_cluster, timestamp)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_cluster_analysis_file(filename)

def load_cluster_analysis_data(B, gamma, mu):
    """Load cluster analysis data from multiple simulation files across different L values."""
    pattern = f"/nbi/nbicmplx/cell/rpw391/babel2D/orderParamL/outputs/L_*_B_{B}/g_{gamma}_a_*_mu_{mu}_*.tsv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    print(f"Found {len(files)} files matching B={B}, gamma={gamma}, mu={mu}")

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
    Aggregate multiple simulations for each (L, alpha) combination.
    Returns: List of (L, alpha, mean_num_clusters, std_error_num, mean_largest_cluster, std_error_largest)
    """
    # Group by (L, alpha)
    grouped_data = {}
    for L, B, gamma, alpha, mu, num_clusters, largest_cluster, timestamp in simulation_data:
        key = (L, alpha)
        if key not in grouped_data:
            grouped_data[key] = {'num_clusters': [], 'largest_cluster': []}
        grouped_data[key]['num_clusters'].append(num_clusters)
        grouped_data[key]['largest_cluster'].append(largest_cluster)

    # Calculate mean and standard error for each group
    aggregated_results = []
    for (L, alpha), data in grouped_data.items():
        num_clusters_array = np.array(data['num_clusters'])
        largest_cluster_array = np.array(data['largest_cluster'])

        mean_num_clusters = np.mean(num_clusters_array)
        std_error_num = np.std(num_clusters_array, ddof=1) / np.sqrt(len(num_clusters_array)) if len(num_clusters_array) > 1 else 0

        mean_largest_cluster = np.mean(largest_cluster_array)
        std_error_largest = np.std(largest_cluster_array, ddof=1) / np.sqrt(len(largest_cluster_array)) if len(largest_cluster_array) > 1 else 0

        aggregated_results.append((L, alpha, mean_num_clusters, std_error_num, mean_largest_cluster,
                                 std_error_largest))

        print(f"L={L}, α={alpha}: {len(num_clusters_array)} sims, num_clusters={mean_num_clusters:.2f}±{std_error_num:.2f}, "
                f"largest_cluster={mean_largest_cluster:.4f}±{std_error_largest:.4f}")

    return aggregated_results

def create_heatmap(aggregated_data, B, gamma, mu, metric_idx, metric_name, colorbar_label, filename_suffix, vmin=None, vmax=None):
    """Create a heatmap for a specific metric over L (y-axis) and alpha (x-axis)."""
    if not aggregated_data:
        print("No data to plot")
        return

    # Convert to arrays for easier handling
    Ls_all = np.array([item[0] for item in aggregated_data])
    alphas_all = np.array([item[1] for item in aggregated_data])
    metric_values = np.array([item[metric_idx] for item in aggregated_data])

    print(f"Successfully processed {len(aggregated_data)} parameter combinations for {metric_name}")

    # Get unique L and alpha values
    Ls = np.sort(np.unique(Ls_all))
    alphas = np.sort(np.unique(alphas_all))

    print(f"L values: {Ls}")
    print(f"Alpha values: {alphas}")

    # Create grid for heatmap (L on y-axis, high to low; alpha on x-axis, low to high)
    Ls_reversed = Ls[::-1]  # High to low for y-axis
    metric_grid = np.full((len(Ls), len(alphas)), np.nan)

    # Fill the grid
    for L, alpha, _, _, _, _ in aggregated_data:
        alpha_idx = np.where(alphas == alpha)[0][0]
        L_idx = np.where(Ls_reversed == L)[0][0]
        # Find the corresponding metric value
        data_idx = np.where((Ls_all == L) & (alphas_all == alpha))[0][0]
        metric_grid[L_idx, alpha_idx] = metric_values[data_idx]

    # Create the heatmap with transparent face color
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='none')
    ax.patch.set_facecolor('none')

    # Set colormap and normalization based on metric
    if metric_name == "Number of Clusters":
        cmap = 'RdBu_r'
        if vmin is None:
            vmin = max(1, np.nanmin(metric_grid))  # Ensure vmin > 0 for log scale
        if vmax is None:
            vmax = np.nanmax(metric_grid)
        # Use log normalization for number of clusters
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:  # Largest Cluster Fraction
        cmap = 'RdBu_r'
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 1
        norm = None

    im = ax.imshow(metric_grid, cmap=cmap, aspect='auto', norm=norm,
                   vmin=vmin if norm is None else None,
                   vmax=vmax if norm is None else None,
                   extent=[alphas.min()-0.1, alphas.max()+0.1,
                          Ls.min()-10, Ls.max()+10])

    # Set ticks and labels - only show min and max
    ax.set_xticks([alphas.min(), alphas.max()])
    ax.set_yticks([Ls.min(), Ls.max()])
    ax.set_xticklabels([f'{alphas.min()}', f'{alphas.max()}'], fontsize=30)
    ax.set_yticklabels([f'{int(Ls.min())}', f'{int(Ls.max())}'], fontsize=30)

    # Customize the plot with matching styling
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=0, fontsize=30)

    # Set colorbar ticks based on metric type
    if metric_name == "Number of Clusters":
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f'{int(vmin)}', f'{int(vmax)}'], fontsize=30)
    else:
        # For largest cluster fraction, show min and max
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0.0', '1.0'], fontsize=30)

    ax.set_xlabel('α', fontsize=40, labelpad=-25)
    ax.set_ylabel('L', fontsize=40, labelpad=-35)

    # Save the plot with transparent background
    output_dir = f"plots/orderParam"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{filename_suffix}_heatmap_B_{B}_g_{gamma}_mu_{mu}.svg"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    print(f"Plot saved to: {fname}")
    plt.close()

def main(B=16, gamma=1.0, mu=0.0001):
    print(f"Looking for files with parameters: B={B}, gamma={gamma}, mu={mu}")

    # Load simulation data
    simulation_data = load_cluster_analysis_data(B, gamma, mu)
    if not simulation_data:
        print("No valid cluster analysis data found.")
        return

    print(f"Successfully loaded data from {len(simulation_data)} simulation files")

    # Aggregate multiple simulations for each parameter combination
    aggregated_data = aggregate_simulations(simulation_data)
    if not aggregated_data:
        print("No valid aggregated data.")
        return

    print(f"Successfully aggregated {len(aggregated_data)} parameter combinations")

    # Create heatmap for number of clusters
    create_heatmap(aggregated_data, B, gamma, mu,
                   metric_idx=2,
                   metric_name="Number of Clusters",
                   colorbar_label="Number of Clusters",
                   filename_suffix="number_of_clusters")

    # Create heatmap for largest cluster fraction
    create_heatmap(aggregated_data, B, gamma, mu,
                   metric_idx=4,
                   metric_name="Largest Cluster Fraction",
                   colorbar_label="Largest Cluster Fraction",
                   filename_suffix="largest_cluster_fraction")

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
