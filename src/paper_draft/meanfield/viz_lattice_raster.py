import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
import multiprocessing as mp
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, mu, simNo from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    sim = re.search(r'_sim_(\d+)\.tsv$', filename)
    if gamma and alpha and mu:
        simNo = int(sim.group(1)) if sim else None
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)), simNo)
    return (None,)*4

def process_order_param_file(filename):
    """
    Process a single meanfield order parameter file to compute time-averaged metrics and their errors.
    Columns: generation, num_languages, largest_cluster_size
    Returns: (gamma, alpha, mu, simNo, mean_num_languages, std_err_num, mean_largest_cluster, std_err_largest)
    """
    gamma, alpha, mu, simNo = extract_params_from_filename(filename)
    if gamma is None or alpha is None or mu is None:
        return None
    try:
        df = pd.read_csv(filename, sep='\t')
        # Handle header or no header
        if 'num_languages' in df.columns and 'largest_cluster_size' in df.columns:
            num_languages = df['num_languages'].values
            largest_cluster = df['largest_cluster_size'].values
        else:
            num_languages = df.iloc[:, 1].values
            largest_cluster = df.iloc[:, 2].values
        mean_num_languages = np.mean(num_languages)
        std_err_num = np.std(num_languages, ddof=1) / np.sqrt(len(num_languages)) if len(num_languages) > 1 else 0
        mean_largest_cluster = np.mean(largest_cluster)
        std_err_largest = np.std(largest_cluster, ddof=1) / np.sqrt(len(largest_cluster)) if len(largest_cluster) > 1 else 0
        return (gamma, alpha, mu, simNo, mean_num_languages, std_err_num, mean_largest_cluster, std_err_largest)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    return process_order_param_file(filename)

def load_order_param_data_gamma_alpha(N, B, mu=None, raster_folder="rasterGamma", gamma=None):
    """
    Load meanfield order parameter data from multiple simulation files for each parameter combination.
    For rasterGamma: filter by N, B, mu.
    For rasterMu: filter by N, B, gamma.
    """
    if raster_folder == "rasterMu":
        # For rasterMu, filter by gamma (mu varies)
        if gamma is None:
            raise ValueError("gamma must be provided for rasterMu")
        pattern = f"src/paper_draft/meanfield/outputs/{raster_folder}/orderParams/N_{N}_B_{B}/g_{gamma}_a_*_mu_*_sim_*.tsv"
    else:
        # Default: rasterGamma, filter by mu (gamma varies)
        if mu is None:
            raise ValueError("mu must be provided for rasterGamma")
        pattern = f"src/paper_draft/meanfield/outputs/{raster_folder}/orderParams/N_{N}_B_{B}/g_*_a_*_mu_{mu}_sim_*.tsv"

    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return []
    print(f"Found {len(files)} files in {raster_folder}/orderParams/N_{N}_B_{B} folder")

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

def aggregate_simulations(simulation_data, raster_folder):
    """
    Aggregate multiple simulations for each parameter combination.
    For rasterGamma: group by (gamma, alpha).
    For rasterMu: group by (mu, alpha).
    Returns: List of (x, alpha, mean_num_languages, std_error_num, mean_largest_cluster, std_error_largest)
    where std_error_* is the propagated error across sims (sqrt(sum(err^2))/n if all errors are independent).
    """
    grouped_data = {}
    for gamma, alpha, mu, simNo, mean_num, err_num, mean_largest, err_largest in simulation_data:
        if raster_folder == "rasterMu":
            key = (mu, alpha)
        else:
            key = (gamma, alpha)
        if key not in grouped_data:
            grouped_data[key] = {'means_num': [], 'errs_num': [], 'means_largest': [], 'errs_largest': []}
        grouped_data[key]['means_num'].append(mean_num)
        grouped_data[key]['errs_num'].append(err_num)
        grouped_data[key]['means_largest'].append(mean_largest)
        grouped_data[key]['errs_largest'].append(err_largest)

    aggregated_results = []
    for key, data in grouped_data.items():
        means_num = np.array(data['means_num'])
        errs_num = np.array(data['errs_num'])
        means_largest = np.array(data['means_largest'])
        errs_largest = np.array(data['errs_largest'])
        n = len(means_num)

        mean_num_languages = np.mean(means_num)
        # Propagate error: sqrt(sum(err^2))/n for per-file errors, plus std error of means
        propagated_err_num = np.sqrt(np.sum(errs_num**2)) / n
        std_err_num = np.std(means_num, ddof=1) / np.sqrt(n) if n > 1 else 0
        total_err_num = np.sqrt(propagated_err_num**2 + std_err_num**2)

        mean_largest_cluster = np.mean(means_largest)
        propagated_err_largest = np.sqrt(np.sum(errs_largest**2)) / n
        std_err_largest = np.std(means_largest, ddof=1) / np.sqrt(n) if n > 1 else 0
        total_err_largest = np.sqrt(propagated_err_largest**2 + std_err_largest**2)

        aggregated_results.append((key[0], key[1], mean_num_languages, total_err_num, mean_largest_cluster, total_err_largest))

        print(f"{'μ' if raster_folder == 'rasterMu' else 'γ'}={key[0]}, α={key[1]}: {n} sims, num_languages={mean_num_languages:.2f}±{total_err_num:.2f}, "
              f"largest_cluster={mean_largest_cluster:.2f}±{total_err_largest:.2f}")

    return aggregated_results

def create_heatmap(aggregated_data, N, B, mu, metric_idx, metric_name, colorbar_label, filename_suffix, raster_folder, vmin=None, vmax=None, lognorm=False):
    """Create a heatmap for a specific metric."""
    if not aggregated_data:
        print("No data to plot")
        return

    # Extract all parameter values
    gammas_all = np.array([item[0] for item in aggregated_data])
    alphas_all = np.array([item[1] for item in aggregated_data])
    mus_all = np.array([item[2] for item in aggregated_data])
    metric_values = np.array([item[metric_idx] for item in aggregated_data])

    print(f"Successfully processed {len(aggregated_data)} parameter combinations for {metric_name}")

    # Determine axes based on raster_folder
    if raster_folder == "rasterMu":
        # x-axis: mu, y-axis: alpha
        mus_axis = np.array([item[0] for item in aggregated_data])    # mu
        alphas_axis = np.array([item[1] for item in aggregated_data]) # alpha
        xs = np.sort(np.unique(mus_axis))
        ys = np.sort(np.unique(alphas_axis))
        xs_label = "μ"
        ys_label = "α"
        xs_reversed = xs  # mu: low to high
        ys_reversed = ys[::-1]  # alpha: high to low
        metric_grid = np.full((len(ys), len(xs)), np.nan)
        for i, alpha in enumerate(ys_reversed):
            for j, mu_val in enumerate(xs_reversed):
                idx = np.where((alphas_axis == alpha) & (mus_axis == mu_val))[0]
                if len(idx) > 0:
                    metric_grid[i, j] = metric_values[idx[0]]
        # Only show leftmost and rightmost ticks for mu axis
        xticks = [0, len(xs)-1]
        xticklabels = [f'{xs[0]:.1e}' if (abs(xs[0]) < 0.01 or abs(xs[0]) > 1000) else f'{xs[0]:g}',
                       f'{xs[-1]:.1e}' if (abs(xs[-1]) < 0.01 or abs(xs[-1]) > 1000) else f'{xs[-1]:g}']
        yticks = [0, len(ys)-1]
        yticklabels = [f'{ys_reversed[0]:.2g}', f'{ys_reversed[-1]:.2g}']
    else:
        # Default: x-axis gamma, y-axis alpha
        xs = np.sort(np.unique(gammas_all))
        ys = np.sort(np.unique(alphas_all))
        xs_label = "γ"
        ys_label = "α"
        xs_reversed = xs  # gamma: low to high
        ys_reversed = ys[::-1]  # alpha: high to low
        metric_grid = np.full((len(ys), len(xs)), np.nan)
        for i, alpha in enumerate(ys_reversed):
            for j, gamma in enumerate(xs_reversed):
                idx = np.where((alphas_all == alpha) & (gammas_all == gamma))[0]
                if len(idx) > 0:
                    metric_grid[i, j] = metric_values[idx[0]]
        xticks = [0, len(xs)-1]
        xticklabels = [f'{xs[0]:.0f}', f'{xs[-1]:.0f}']
        yticks = [0, len(ys)-1]
        yticklabels = [f'{ys_reversed[0]:.0f}', f'{ys_reversed[-1]:.0f}']

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='none')
    ax.patch.set_facecolor('none')

    cmap = 'RdBu_r'
    if vmin is None:
        vmin = np.nanmin(metric_grid)
    if vmax is None:
        vmax = np.nanmax(metric_grid)
    norm = LogNorm(vmin=vmin, vmax=vmax) if lognorm else None

    im = ax.imshow(metric_grid, cmap=cmap, aspect='auto', norm=norm)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, fontsize=30)
    ax.set_yticklabels(yticklabels, fontsize=30)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, rotation=270, labelpad=-50, fontsize=30)
    cbar.set_ticks([vmin, vmax])
    # Round colorbar tick labels to integers
    cbar.set_ticklabels([f'{int(round(vmin))}', f'{int(round(vmax))}'], fontsize=30)

    ax.set_xlabel(xs_label, fontsize=40, labelpad=-25)
    ax.set_ylabel(ys_label, fontsize=40, labelpad=-35)

    output_dir = f"src/paper_draft/meanfield/plots/{raster_folder}/N_{N}_B_{B}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/{filename_suffix}_heatmap_N_{N}_B_{B}_mu_{mu}.pdf"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    print(f"Plot saved to: {fname}")
    plt.close()

def main_meanfield_order_param_gamma_alpha(N=65536, B=16, mu=0.0001, raster_folder="rasterGamma", gamma=None):
    print(f"Looking for files with parameters: N={N}, B={B}, mu={mu}, gamma={gamma} in {raster_folder}/orderParams folder")

    if raster_folder == "rasterMu":
        simulation_data = load_order_param_data_gamma_alpha(N, B, mu=None, raster_folder=raster_folder, gamma=gamma)
    else:
        simulation_data = load_order_param_data_gamma_alpha(N, B, mu=mu, raster_folder=raster_folder, gamma=None)

    if not simulation_data:
        print("No valid order parameter data found.")
        return

    print(f"Successfully loaded data from {len(simulation_data)} simulation files")

    aggregated_data = aggregate_simulations(simulation_data, raster_folder)
    if not aggregated_data:
        print("No valid aggregated data.")
        return

    print(f"Successfully aggregated {len(aggregated_data)} parameter combinations")

    # Heatmap for number of languages (clusters)
    create_heatmap(aggregated_data, N, B, mu if raster_folder != "rasterMu" else "varied",
                   metric_idx=2,
                   metric_name="Number of Languages",
                   colorbar_label="Number of Languages",
                   filename_suffix=f"num_languages_{raster_folder}",
                   raster_folder=raster_folder,
                   lognorm=True)

    # Heatmap for largest cluster size
    create_heatmap(aggregated_data, N, B, mu if raster_folder != "rasterMu" else "varied",
                   metric_idx=4,
                   metric_name="Largest Cluster Size",
                   colorbar_label="Largest Cluster Size",
                   filename_suffix=f"largest_cluster_size_{raster_folder}",
                   raster_folder=raster_folder,
                   lognorm=False)

if __name__ == "__main__":
    import sys

    # Default parameters
    N = 65536
    B = 16

    # mu = 0.0001
    # raster_folder = "rasterGamma"
    # gamma = None

    mu = None
    raster_folder = "rasterMu"
    gamma = 1


    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        B = int(sys.argv[2])
    if len(sys.argv) > 3:
        mu = float(sys.argv[3])
    if len(sys.argv) > 4:
        raster_folder = sys.argv[4]
    if len(sys.argv) > 5:
        gamma = float(sys.argv[5])

    main_meanfield_order_param_gamma_alpha(N, B, mu, raster_folder, gamma)
    if len(sys.argv) > 5:
        gamma = float(sys.argv[5])

    main_meanfield_order_param_gamma_alpha(N, B, mu, raster_folder, gamma)
