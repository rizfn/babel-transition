import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import multiprocessing as mp
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+)', filename)
    
    if gamma and alpha and L and B and mu:
        return (float(gamma.group(1)), float(alpha.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)))
    return (None,)*5

def parse_cluster_line(line):
    """Parse a line from cluster timeseries file."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    
    step = int(parts[0])
    
    # Parse cluster data for each language
    all_cluster_sizes = []
    for i in range(1, len(parts)):
        lang_data = parts[i]
        if ':' in lang_data:
            lang, sizes_str = lang_data.split(':', 1)
            if sizes_str:  # Only process if there are cluster sizes
                sizes = [int(s) for s in sizes_str.split(',')]
                all_cluster_sizes.extend(sizes)
    
    return step, all_cluster_sizes

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
    Process a single cluster timeseries file to compute sqrt of weighted average cluster size and its SEM.
    """
    gamma, alpha, L, B, mu = extract_params_from_filename(filename)
    if alpha is None:
        return None

    # Store all cluster sizes for error propagation
    all_cluster_sizes = []

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                step, cluster_sizes = parse_cluster_line(line)
                if cluster_sizes is None or len(cluster_sizes) == 0:
                    continue
                all_cluster_sizes.extend(cluster_sizes)

        if len(all_cluster_sizes) == 0:
            sqrt_weighted_cluster_size = 0.0
            sem_sqrt_weighted = 0.0
        else:
            cluster_sizes = np.array(all_cluster_sizes)
            total_area = np.sum(cluster_sizes)
            weighted_sum = np.sum(cluster_sizes**2)
            if total_area == 0:
                sqrt_weighted_cluster_size = 0.0
                sem_sqrt_weighted = 0.0
            else:
                expectation_squared = weighted_sum / total_area
                sqrt_weighted_cluster_size = np.sqrt(expectation_squared)

                # Error propagation for sqrt(E[A^2])
                # Var(E[A^2]) = (E[A^4] - (E[A^2])^2) / N
                N = len(cluster_sizes)
                E_A2 = expectation_squared
                E_A4 = np.sum(cluster_sizes**4) / total_area
                var_EA2 = (E_A4 - E_A2**2) / N if N > 1 else 0.0
                # Propagate through sqrt: Var(sqrt(X)) ≈ Var(X) / (4 X)
                if E_A2 > 0 and var_EA2 > 0:
                    sem_sqrt_weighted = np.sqrt(var_EA2) / (2 * np.sqrt(E_A2))
                else:
                    sem_sqrt_weighted = 0.0

        return (gamma, alpha, L, B, mu, sqrt_weighted_cluster_size, sem_sqrt_weighted)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_memory_efficient(filename)

def load_cluster_data(L, B, gamma, mu, output_dir="constantGamma"):
    """Load cluster data from files matching the specified parameters."""
    # Get the current directory (stochasticCommutable)
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/clusterTimeseries/{output_dir}/L_{L}_g_{gamma}_a_*_B_{B}_mu_{mu}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files matching L={L}, B={B}, gamma={gamma}, mu={mu}")
    
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
                    gamma_file, alpha, L_file, B_file, mu_file, sqrt_weighted_cluster_size, sem_sqrt_weighted = result
                    
                    # Double-check that the extracted parameters match what we're looking for
                    if (L_file == L and B_file == B and gamma_file == gamma and 
                        mu_file == mu):
                        results.append(result)
                
                pbar.update(1)
    
    return results

def main(L=128, B=16, gamma=3, mu=0.001, output_dir="constantGamma"):
    """Main function that takes parameters and finds matching files."""
    print(f"Looking for files with parameters: L={L}, B={B}, gamma={gamma}, mu={mu}")
    
    # Load cluster data from files matching the specified parameters
    results = load_cluster_data(L, B, gamma, mu, output_dir)
    
    if len(results) == 0:
        print("No valid cluster data found.")
        return
    
    print(f"Successfully processed {len(results)} files")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results, columns=['gamma', 'alpha', 'L', 'B', 'mu', 'sqrt_weighted_cluster_size', 'sem_sqrt_weighted'])

    grouped = df.groupby('alpha')
    means = grouped['sqrt_weighted_cluster_size'].mean()
    combined_sem = grouped['sem_sqrt_weighted'].apply(lambda x: np.sqrt(np.sum(x**2)) / len(x))

    alphas = means.index.values
    mean_vals = means.values
    sem_vals = combined_sem.values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, mean_vals, 'o-', color='blue', linewidth=2, markersize=6, label='Mean')
    ax.fill_between(alphas, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.3, color='blue', label='SEM')

    ax.set_xlabel('Alpha (local interaction strength)', fontsize=12)
    ax.set_ylabel('√(E[Cluster Size²])', fontsize=12)
    ax.set_title(f'Square Root of Weighted Cluster Size vs Alpha\n(L={L}, B={B}, γ={gamma}, μ={mu})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_plot_dir = f"src/understandabilityVsHamming2D/stochasticCommutable/plots/clusterSizes/{output_dir}"
    os.makedirs(output_plot_dir, exist_ok=True)
    fname = f"{output_plot_dir}/sqrt_weighted_cluster_size_vs_alpha_L_{L}_B_{B}_g_{gamma}_mu_{mu}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')

    print(f"Plot saved to: {fname}")


def process_file_mean_cluster_size(filename):
    """
    Process a single cluster timeseries file to compute mean cluster size and its SEM.
    """
    gamma, alpha, L, B, mu = extract_params_from_filename(filename)
    if alpha is None:
        return None

    all_cluster_sizes = []

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                step, cluster_sizes = parse_cluster_line(line)
                if cluster_sizes is None or len(cluster_sizes) == 0:
                    continue
                all_cluster_sizes.extend(cluster_sizes)

        if len(all_cluster_sizes) == 0:
            mean_cluster_size = 0.0
            sem_cluster_size = 0.0
        else:
            mean_cluster_size = np.mean(all_cluster_sizes)
            sem_cluster_size = np.std(all_cluster_sizes, ddof=1) / np.sqrt(len(all_cluster_sizes))

        return (gamma, alpha, L, B, mu, mean_cluster_size, sem_cluster_size)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file_mean(filename):
    """Wrapper for multiprocessing mean cluster size."""
    return process_file_mean_cluster_size(filename)

def main_mean_cluster_size(L=128, B=16, gamma=3, mu=0.001, output_dir="constantGamma"):
    """Main function for mean cluster size vs alpha."""
    print(f"Looking for files with parameters: L={L}, B={B}, gamma={gamma}, mu={mu}")

    # Load cluster data from files matching the specified parameters
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/clusterTimeseries/{output_dir}/L_{L}_g_{gamma}_a_*_B_{B}_mu_{mu}.tsv")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}

    print(f"Found {len(files)} files matching L={L}, B={B}, gamma={gamma}, mu={mu}")

    # Determine number of processes (leave 4 CPUs free)
    total_cpus = mp.cpu_count()
    num_processes = max(1, total_cpus - 4)
    print(f"Using {num_processes} processes (out of {total_cpus} CPUs)")

    # Process files in parallel
    results = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=len(files), desc="Processing files (mean cluster size)") as pbar:
            for result in pool.imap(worker_process_file_mean, files):
                if result is not None:
                    gamma_file, alpha, L_file, B_file, mu_file, mean_cluster_size, sem_cluster_size = result
                    if (L_file == L and B_file == B and gamma_file == gamma and mu_file == mu):
                        results.append(result)
                pbar.update(1)

    if len(results) == 0:
        print("No valid cluster data found.")
        return

    print(f"Successfully processed {len(results)} files")

    # After collecting results:
    df = pd.DataFrame(results, columns=['gamma', 'alpha', 'L', 'B', 'mu', 'mean_cluster_size', 'sem_cluster_size'])

    grouped = df.groupby('alpha')
    means = grouped['mean_cluster_size'].mean()
    # Combine SEMs in quadrature and divide by n (number of files for that alpha)
    combined_sem = grouped['sem_cluster_size'].apply(lambda x: np.sqrt(np.sum(x**2)) / len(x))

    alphas = means.index.values
    mean_vals = means.values
    sem_vals = combined_sem.values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, mean_vals, 'o-', color='green', linewidth=2, markersize=6, label='Mean Cluster Size')
    ax.fill_between(alphas, mean_vals - sem_vals, mean_vals + sem_vals, color='green', alpha=0.2, label='SEM')

    ax.set_xlabel('Alpha (local interaction strength)', fontsize=12)
    ax.set_ylabel('Mean Cluster Size', fontsize=12)
    ax.set_title(f'Mean Cluster Size vs Alpha\n(L={L}, B={B}, γ={gamma}, μ={mu})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save the plot in the current directory's plots folder
    output_plot_dir = f"src/understandabilityVsHamming2D/stochasticCommutable/plots/clusterSizes/{output_dir}"
    os.makedirs(output_plot_dir, exist_ok=True)
    fname = f"{output_plot_dir}/mean_cluster_size_vs_alpha_L_{L}_B_{B}_g_{gamma}_mu_{mu}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')

    print(f"Plot saved to: {fname}")

if __name__ == "__main__":
    main(L=128, B=16, gamma=3, mu=0.001, output_dir="constantGamma")
    # main_mean_cluster_size(L=128, B=16, gamma=3, mu=0.001, output_dir="constantGamma")