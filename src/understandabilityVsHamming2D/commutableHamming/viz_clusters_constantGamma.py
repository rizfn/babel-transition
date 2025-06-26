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
import argparse

def extract_alpha_from_filename(filename):
    """Extract alpha from filename."""
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    if alpha:
        return float(alpha.group(1))
    return None

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
    alpha = extract_alpha_from_filename(filename)
    
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
        
        return (alpha, sqrt_weighted_cluster_size)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def worker_process_file(filename):
    """Wrapper function for multiprocessing."""
    return process_file_memory_efficient(filename)

def main():
    parser = argparse.ArgumentParser(description='Analyze cluster timeseries data for varying alpha values')
    parser.add_argument('--L', type=int, default=256, help='Lattice size')
    parser.add_argument('--B', type=int, default=16, help='Bitstring length')
    parser.add_argument('--gamma', type=float, default=3, help='Gamma value')
    parser.add_argument('--mu', type=float, default=0.001, help='Mutation rate')
    parser.add_argument('--K', type=int, default=1, help='Kill radius')
    parser.add_argument('--output_dir', type=str, default="constantGamma", help='Output subdirectory')
    
    args = parser.parse_args()
    
    # Build file pattern with wildcard for alpha
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/clusterTimeseries/{args.output_dir}/L_{args.L}_g_{args.gamma}_a_*_B_{args.B}_mu_{args.mu}_K_{args.K}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files")
    print(f"Parameters: L={args.L}, B={args.B}, gamma={args.gamma}, mu={args.mu}, K={args.K}")
    
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
                    results.append(result)
                pbar.update(1)
    
    if len(results) == 0:
        print("No valid results found.")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results, columns=['alpha', 'sqrt_weighted_cluster_size'])
    
    print(f"Successfully processed {len(df)} files")
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
    ax.set_title(f'Square Root of Weighted Cluster Size vs Alpha\n(γ={args.gamma}, L={args.L}, B={args.B}, μ={args.mu}, K={args.K})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the plot
    output_dir = f"src/understandabilityVsHamming2D/commutableHamming/plots/clusterSizes/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/sqrt_weighted_cluster_size_vs_alpha_g_{args.gamma}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")

if __name__ == "__main__":
    main()