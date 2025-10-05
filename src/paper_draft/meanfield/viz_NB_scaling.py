import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Set this to the outputs directory where the NBScaling results are stored
BASE_DIR = "src/paper_draft/meanfield/outputs/NBScaling/"
DEFAULT_N = 65536
DEFAULT_B = 16
DEFAULT_GAMMA = 1
DEFAULT_MU = 0.0001

def extract_params_from_filename(filename):
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    sim = re.search(r'_sim_(\d+)\.tsv$', filename)
    if gamma and alpha and mu and sim:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)), int(sim.group(1)))
    return (None,)*4

def process_order_param_file(filename):
    gamma, alpha, mu, sim = extract_params_from_filename(filename)
    if gamma is None or alpha is None or mu is None or sim is None:
        return None
    # Only process files with gamma=DEFAULT_GAMMA (ignore mu)
    if gamma != DEFAULT_GAMMA:
        return None
    try:
        df = pd.read_csv(filename, sep='\t')
        if 'num_languages' in df.columns and 'largest_cluster_size' in df.columns:
            num_clusters_values = df['num_languages'].values
            largest_cluster_values = df['largest_cluster_size'].values
        else:
            num_clusters_values = df.iloc[:, 1].values
            largest_cluster_values = df.iloc[:, 2].values
        mean_num_clusters = np.mean(num_clusters_values)
        mean_largest_cluster = np.mean(largest_cluster_values)
        return (alpha, mean_num_clusters, mean_largest_cluster)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def aggregate_by_alpha(data):
    grouped = {}
    for alpha, num_clusters, largest_cluster in data:
        if alpha not in grouped:
            grouped[alpha] = {'num_clusters': [], 'largest_cluster': []}
        grouped[alpha]['num_clusters'].append(num_clusters)
        grouped[alpha]['largest_cluster'].append(largest_cluster)
    alphas = sorted(grouped.keys())
    agg = {'alpha': [], 'num_clusters_mean': [], 'num_clusters_err': [],
           'largest_cluster_mean': [], 'largest_cluster_err': []}
    for alpha in alphas:
        agg['alpha'].append(alpha)
        for key, mean_key, err_key in [
            ('num_clusters', 'num_clusters_mean', 'num_clusters_err'),
            ('largest_cluster', 'largest_cluster_mean', 'largest_cluster_err')
        ]:
            vals = grouped[alpha][key]
            agg[mean_key].append(np.mean(vals))
            agg[err_key].append(np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0)
    return agg

def parse_N_B_from_folder(foldername):
    m = re.match(r'N_(\d+)_B_(\d+)', foldername)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def get_all_sim_folders(base_dir):
    folders = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            N, B = parse_N_B_from_folder(entry)
            if N is not None and B is not None:
                folders.append((N, B, full_path))
    return folders

def round_to_n_sig_figs(x, n=2):
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

def plot_scaling(all_results, param_name, ylabel, filename, divide_by_N=False):
    plt.figure(figsize=(10, 8), facecolor='none')
    ax = plt.gca()
    ax.patch.set_facecolor('none')

    # Color/marker scheme
    color_default = 'k'
    marker_default = 'o'
    marker_B = 'D'
    marker_N = 's'
    linewidth = 2

    # Prepare color maps for B and N
    # Get all unique B (for N fixed) and N (for B fixed), excluding default
    Bs = sorted({B for (N, B) in all_results if N == DEFAULT_N and B != DEFAULT_B})
    Ns = sorted({N for (N, B) in all_results if N != DEFAULT_N and B == DEFAULT_B})

    reds = plt.cm.Reds(np.linspace(0.5, 0.9, len(Bs))) if Bs else []
    blues = plt.cm.Blues(np.linspace(0.5, 0.9, len(Ns))) if Ns else []

    B_color_map = {b: reds[i] for i, b in enumerate(Bs)}
    N_color_map = {n: blues[i] for i, n in enumerate(Ns)}

    for (N, B), agg in all_results.items():
        alphas = np.array(agg['alpha'])
        y = np.array(agg[f'{param_name}_mean'])
        yerr = np.array(agg[f'{param_name}_err'])
        if divide_by_N:
            y = y / N
            yerr = yerr / N

        if N == DEFAULT_N and B == DEFAULT_B:
            label = f'Default N={N}, B={B}'
            color = color_default
            marker = marker_default
            zorder = 10
        elif N == DEFAULT_N and B != DEFAULT_B:
            label = f'N={N}, B={B}'
            color = B_color_map.get(B, 'tab:orange')
            marker = marker_B
            zorder = 5
        elif N != DEFAULT_N and B == DEFAULT_B:
            label = f'N={N}, B={B}'
            color = N_color_map.get(N, 'tab:blue')
            marker = marker_N
            zorder = 5
        else:
            continue  # skip cases where both N and B differ from default
        ax.errorbar(alphas, y, yerr=yerr, label=label, color=color, marker=marker, linewidth=linewidth, markersize=8, capsize=4, zorder=zorder)

    ax.set_xlabel('α', fontsize=40, labelpad=-25)
    ax.set_ylabel(ylabel, fontsize=40)  # removed labelpad

    ax.set_ylim(bottom=0)  # Always start y-axis at 0

    # Set x ticks to min/max alpha
    all_alphas_plot = np.concatenate([np.array(agg['alpha']) for agg in all_results.values()])
    ax.set_xticks([all_alphas_plot.min(), all_alphas_plot.max()])
    ax.set_xticklabels([f'{all_alphas_plot.min():.2g}', f'{all_alphas_plot.max():.2g}'], fontsize=30)
    ax.tick_params(axis='y', labelsize=30)

    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax.minorticks_on()
    ax.legend(fontsize=20)
    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "plots/NBScaling")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}.pdf", dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    print(f"Plot saved to: {output_dir}/{filename}.pdf")
    plt.close()

def main():
    folders = get_all_sim_folders(BASE_DIR)
    all_results = {}
    for N, B, folder in folders:
        # Match any mu value
        files = glob.glob(os.path.join(folder, f"g_{DEFAULT_GAMMA}_a_*_mu_*_sim_*.tsv"))
        if not files:
            print(f"Warning: No files found in {folder} matching pattern g_{DEFAULT_GAMMA}_a_*_mu_*_sim_*.tsv")
        data = []
        for f in files:
            result = process_order_param_file(f)
            if result is not None:
                data.append(result)
        if data:
            agg = aggregate_by_alpha(data)
            all_results[(N, B)] = agg

    # Only keep (N, B) where at most one differs from default
    filtered_results = {}
    for (N, B), agg in all_results.items():
        if (N == DEFAULT_N and B == DEFAULT_B) or (N == DEFAULT_N and B != DEFAULT_B) or (N != DEFAULT_N and B == DEFAULT_B):
            filtered_results[(N, B)] = agg

    if not filtered_results:
        print("No data found for any (N, B) combination. Please check BASE_DIR and file patterns.")
        return

    # Plot for each order parameter
    plot_scaling(filtered_results, 'num_clusters', "Number of Clusters", "NB_scaling_num_clusters")
    plot_scaling(filtered_results, 'num_clusters', "Number of Clusters", "NB_scaling_num_clusters_per_site", divide_by_N=True)
    plot_scaling(filtered_results, 'largest_cluster', "Largest Cluster", "NB_scaling_largest_cluster_fraction", divide_by_N=True)

if __name__ == "__main__":
    main()
