import numpy as np
import matplotlib.pyplot as plt
import cc3d
import os
import argparse

def parse_lattice_line(line, L, B):
    """Parse a single line from the lattice timeseries format."""
    step, lattice_str = line.strip().split('\t')
    rows = lattice_str.split(';')
    lattice = np.zeros((L, L, B), dtype=int)
    for i, row in enumerate(rows):
        cells = row.split(',')
        for j, cell in enumerate(cells):
            bits = [int(b) for b in cell]
            lattice[i, j, :] = bits
    return int(step), lattice

def lattice_to_language_grid(lattice):
    """Convert lattice of bitstrings to grid of language IDs."""
    L, _, B = lattice.shape
    # Convert bitstrings to integers using numpy's binary interpretation
    powers = 2 ** np.arange(B)
    lang_grid = np.sum(lattice * powers, axis=2)
    lang_grid = lang_grid.reshape(L, L).astype(int) + 1  # +1 to avoid zero language (background for cc3d)
    return lang_grid


def find_clusters_periodic(lang_grid):
    """Find connected clusters with periodic boundary conditions."""
    labels = cc3d.connected_components(lang_grid, connectivity=4, periodic_boundary=True)
    
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    
    return cluster_sizes

def analyze_cluster_size_distribution(L, B, gamma, alpha, r, mu, K, bins=50, start_step=500):
    """Analyze cluster size distribution across all time steps, starting from start_step."""
    filename = f"src/understandabilityVsHammingSmall2D/outputs/latticeNbrVsGlobalTimeseries/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}.tsv"
    
    all_cluster_sizes = []
    
    with open(filename, "r") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            step, lattice = parse_lattice_line(line, L, B)
            
            # Skip transient period
            if step < start_step:
                continue
                
            lang_grid = lattice_to_language_grid(lattice)
            cluster_sizes = find_clusters_periodic(lang_grid)
            all_cluster_sizes.extend(cluster_sizes)
            
            if line_num % 10 == 0:
                print(f"Processed step {step}, found {len(cluster_sizes)} clusters")
    
    if not all_cluster_sizes:
        print("No clusters found!")
        return
    
    # Create arrays
    cluster_sizes_array = np.array(all_cluster_sizes)
    min_size = np.min(cluster_sizes_array)
    max_size = np.max(cluster_sizes_array)
    
    # Linear-spaced bins for log-linear plot
    linear_bins = np.linspace(min_size, max_size, bins)
    linear_counts, linear_bin_edges = np.histogram(cluster_sizes_array, bins=linear_bins)
    linear_bin_centers = (linear_bin_edges[:-1] + linear_bin_edges[1:]) / 2
    
    # Log-spaced bins for log-log plot
    log_bins = np.geomspace(min_size, max_size, bins)
    log_counts, log_bin_edges = np.histogram(cluster_sizes_array, bins=log_bins)
    log_bin_centers = (log_bin_edges[:-1] + log_bin_edges[1:]) / 2
    log_bin_widths = np.diff(log_bin_edges)
    log_density = log_counts / log_bin_widths  # Normalize by bin width
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Log-Linear scale histogram (linear bins, log y-axis)
    ax1.bar(linear_bin_centers, linear_counts, width=np.diff(linear_bin_edges), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Cluster Size')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_title('Cluster Size Distribution (Log-linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale histogram (log bins, log axes, normalized by bin width)
    ax2.bar(log_bin_centers, log_density, width=log_bin_widths, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Frequency / Bin Width')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Cluster Size Distribution (Log-Log Scale, Density)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Cluster Size Distribution (Steps {start_step}+)\nL={L}, B={B}, γ={gamma}, α={alpha}, r={r}, μ={mu}, K={K}')
    plt.tight_layout()
    
    # Save plot
    outdir = f"src/understandabilityVsHammingSmall2D/plots/clusterSizeDistribution"
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, f"cluster_dist_L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}_start{start_step}.png")
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--r", type=float, default=2)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--start_step", type=int, default=500, help="Step to start analysis from")
    args = parser.parse_args()

    analyze_cluster_size_distribution(args.L, args.B, args.gamma, args.alpha, args.r, args.mu, args.K, args.bins, args.start_step)
