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
    Process a single cluster timeseries file to compute sqrt of weighted average cluster size.
    """
    gamma, alpha, L, B, mu, K = extract_params_from_filename(filename)
    
    if alpha is None:
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
                
                # Parse this single line
                step, cluster_sizes = parse_cluster_line(line)
                if cluster_sizes is None or len(cluster_sizes) == 0:
                    continue
                
                # Accumulate statistics
                cluster_sizes = np.array(cluster_sizes)
                total_area_sum += np.sum(cluster_sizes)
                total_weighted_sum += np.sum(cluster_sizes**2)
        
        # Compute sqrt of weighted average cluster size across all timesteps
        if total_area_sum == 0:
            sqrt_weighted_cluster_size = 0.0
        else:
            expectation_squared = total_weighted_sum / total_area_sum
            sqrt_weighted_cluster_size = np.sqrt(expectation_squared)
        
        return (gamma, alpha, L, B, mu, K, sqrt_weighted_cluster_size)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_memory_efficient(filename)

def load_cluster_data(L, B, gamma, mu, K, output_dir="constantGamma"):
    """Load cluster data from files matching the specified parameters."""
    # First, find all files in the directory
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/clusterTimeseries/{output_dir}/L_{L}_g_{gamma}_a_*_B_{B}_mu_{mu}_K_{K}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files matching L={L}, B={B}, gamma={gamma}, mu={mu}, K={K}")
    
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
                    gamma_file, alpha, L_file, B_file, mu_file, K_file, sqrt_weighted_cluster_size = result
                    
                    # Double-check that the extracted parameters match what we're looking for
                    if (L_file == L and B_file == B and gamma_file == gamma and 
                        mu_file == mu and K_file == K):
                        results.append(result)
                
                pbar.update(1)
    
    return results

def main(L=128, B=16, gamma=3, mu=0.001, K=1, output_dir="constantGamma"):
    """Main function that takes parameters and finds matching files."""
    print(f"Looking for files with parameters: L={L}, B={B}, gamma={gamma}, mu={mu}, K={K}")
    
    # Load cluster data from files matching the specified parameters
    results = load_cluster_data(L, B, gamma, mu, K, output_dir)
    
    if len(results) == 0:
        print("No valid cluster data found.")
        return
    
    print(f"Successfully processed {len(results)} files")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results, columns=['gamma', 'alpha', 'L', 'B', 'mu', 'K', 'sqrt_weighted_cluster_size'])
    
    print("DataFrame summary:")
    print(df.head())
    
    # Group by alpha and compute statistics
    stats = df.groupby('alpha')['sqrt_weighted_cluster_size'].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])  # Standard error of mean
    
    print(f"Alpha values: {sorted(stats['alpha'].values)}")
    
    # Create the line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alphas = stats['alpha'].values
    means = stats['mean'].values
    sems = stats['sem'].values
    
    # Plot mean line
    ax.plot(alphas, means, 'o-', color='blue', linewidth=2, markersize=6, label='Mean')
    
    # Fill 1 sigma error region
    ax.fill_between(alphas, means - sems, means + sems, alpha=0.3, color='blue', label='±1 SEM')
    
    # Customize the plot
    ax.set_xlabel('Alpha (local interaction strength)', fontsize=12)
    ax.set_ylabel('√(E[Cluster Size²])', fontsize=12)
    ax.set_title(f'Square Root of Weighted Cluster Size vs Alpha\n(L={L}, B={B}, γ={gamma}, μ={mu}, K={K})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the plot with all parameters in filename
    output_plot_dir = f"src/understandabilityVsHamming2D/commutableHamming/plots/clusterSizes/{output_dir}"
    os.makedirs(output_plot_dir, exist_ok=True)
    fname = f"{output_plot_dir}/sqrt_weighted_cluster_size_vs_alpha_L_{L}_B_{B}_g_{gamma}_mu_{mu}_K_{K}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")

if __name__ == "__main__":
    main(L=256, B=16, gamma=3, mu=0.001, K=1, output_dir="constantGamma")