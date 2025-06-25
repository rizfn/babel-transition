import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cc3d
import re
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

def parse_lattice_line(line):
    """Parse a single line into a 2D lattice array."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    
    step = int(parts[0])  # First column is the step number
    lattice_data = parts[1]  # Second column contains the lattice
    
    # Parse the lattice data: rows separated by ';', cells by ','
    rows = lattice_data.split(';')
    lattice = []
    for row in rows:
        cells = row.split(',')
        lattice_row = []
        for cell in cells:
            # Each cell is a bitstring, convert to tuple for uniqueness
            bits = tuple(int(b) for b in cell)
            lattice_row.append(bits)
        lattice.append(lattice_row)
    
    # Return as regular list of lists, not numpy array to avoid hashability issues
    return step, lattice

def compute_clusters_cc3d(lattice_2d):
    """
    Compute clusters using cc3d with periodic boundaries.
    Each unique language gets a unique integer ID for cc3d.
    """
    L = len(lattice_2d)  # Use len() instead of .shape since it's now a list
    
    # Create a mapping from unique languages to integer IDs
    unique_languages = set()
    for i in range(L):
        for j in range(L):
            unique_languages.add(lattice_2d[i][j])  # Now accessing as list[i][j]
    
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
    
    # Count unique cluster labels (no need to exclude 0 anymore since we start from 1)
    unique_labels = np.unique(labels)
    cluster_sizes = []
    for label in unique_labels:
        cluster_size = np.sum(labels == label)
        cluster_sizes.append(cluster_size)
    
    return cluster_sizes

def compute_avg_communication_partners(cluster_sizes):
    """
    Compute the average number of communication partners per agent.
    For each agent in a cluster of size A, they can communicate with (A-1) others.
    Returns the average across all agents: Σᵢ Aᵢ(Aᵢ-1) / Σᵢ Aᵢ
    """
    if len(cluster_sizes) == 0:
        return 0.0
    
    cluster_sizes = np.array(cluster_sizes)
    total_agents = np.sum(cluster_sizes)
    total_communication_pairs = np.sum(cluster_sizes * (cluster_sizes - 1))
    
    if total_agents == 0:
        return 0.0
    
    return total_communication_pairs / total_agents

def get_last_line_from_file(filename):
    """Get the last non-empty line from a file."""
    with open(filename, 'rb') as f:
        # Seek to end and work backwards to find last non-empty line
        f.seek(0, 2)  # Go to end of file
        file_size = f.tell()
        
        if file_size == 0:
            return None
            
        # Read backwards to find the last line
        f.seek(-1, 2)
        lines = []
        while f.tell() > 0:
            char = f.read(1)
            if char == b'\n':
                if lines:  # We found a complete line
                    break
            f.seek(-2, 1)  # Move back 2 positions
            lines.append(char)
        
        # Read any remaining content if we reached the beginning
        if f.tell() == 0:
            f.seek(0)
            remaining = f.read().decode('utf-8')
            return remaining.strip().split('\n')[-1]
        
        # Decode the line we found
        last_line = b''.join(reversed(lines)).decode('utf-8').strip()
        return last_line if last_line else None

def load_final_communication_data():
    """Load the final lattice from each file and compute average communication partners."""
    pattern = os.path.join(os.path.dirname(__file__), "outputs/latticeTimeseries/rasterscan/L_*_g_*_a_*_B_*_mu_*_K_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files")
    
    communication_data = {}
    
    for filename in tqdm(files, desc="Processing final lattices"):
        gamma, alpha, L, B, mu, K = extract_params_from_filename(filename)
        
        if gamma is None or alpha is None:
            continue
            
        try:
            last_line = get_last_line_from_file(filename)
            if last_line is None:
                continue
                
            step, lattice = parse_lattice_line(last_line)
            if lattice is not None:
                # Compute cluster sizes for the final lattice
                cluster_sizes = compute_clusters_cc3d(lattice)
                avg_partners = compute_avg_communication_partners(cluster_sizes)
                
                communication_data[(gamma, alpha)] = {
                    'avg_communication_partners': avg_partners,
                    'step': step,
                    'L': L,
                    'B': B,
                    'mu': mu,
                    'K': K,
                    'num_clusters': len(cluster_sizes),
                    'total_agents': sum(cluster_sizes) if cluster_sizes else 0
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return communication_data

def create_communication_heatmap(communication_data):
    """Create a heatmap of average communication partners organized by gamma and alpha."""
    if not communication_data:
        print("No communication data to plot")
        return
    
    # Get unique gamma and alpha values
    gammas = sorted(set(key[0] for key in communication_data.keys()))
    alphas = sorted(set(key[1] for key in communication_data.keys()))
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create grid for heatmap using raw values
    communication_grid = np.full((len(alphas), len(gammas)), np.nan)
    
    # Fill the grid
    for (gamma, alpha), data in communication_data.items():
        gamma_idx = np.where(np.array(gammas) == gamma)[0][0]
        alpha_idx = np.where(np.array(alphas) == alpha)[0][0]
        communication_grid[alpha_idx, gamma_idx] = data['avg_communication_partners']
    
    # Create the heatmap with linear scale
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(communication_grid, cmap='viridis', aspect='auto',
                   extent=[min(gammas)-0.5, max(gammas)+0.5, 
                          min(alphas)-0.5, max(alphas)+0.5],
                   origin='lower')
    
    # Add text annotations
    x_centers = np.linspace(min(gammas), max(gammas), len(gammas))
    y_centers = np.linspace(min(alphas), max(alphas), len(alphas))
    
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            if not np.isnan(communication_grid[i, j]):
                value = communication_grid[i, j]
                # Determine text color based on brightness
                vmin, vmax = np.nanmin(communication_grid), np.nanmax(communication_grid)
                if vmin == vmax:
                    brightness = 0.5
                else:
                    brightness = (value - vmin) / (vmax - vmin)
                text_color = 'white' if brightness > 0.5 else 'black'
                
                ax.text(x_centers[j], y_centers[i], f"{value:.1f}", 
                       ha='center', va='center', 
                       color=text_color, fontweight='bold')
    
    # Customize the plot
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Communication Partners per Agent', rotation=270, labelpad=20)
    
    ax.set_xticks(gammas)
    ax.set_yticks(alphas)
    ax.set_xticks(gammas, minor=True)
    ax.set_yticks(alphas, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Gamma (Global Interaction Strength)')
    ax.set_ylabel('Alpha (Local Interaction Strength)')
    ax.set_title('Average Communication Partners per Agent (Final State)')
    
    # Save the plot
    output_dir = "src/understandabilityVsHamming2D/commutableHamming/plots/clusterSizes"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/heatmap_communication_partners_final.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    print(f"Plot saved to: {fname}")
    
    # Save the results as CSV
    results_list = []
    for (gamma, alpha), data in communication_data.items():
        results_list.append({
            'gamma': gamma,
            'alpha': alpha,
            'avg_communication_partners': data['avg_communication_partners'],
            'step': data['step'],
            'L': data['L'],
            'B': data['B'],
            'mu': data['mu'],
            'K': data['K'],
            'num_clusters': data['num_clusters'],
            'total_agents': data['total_agents']
        })
    
    df = pd.DataFrame(results_list)
    csv_fname = f"{output_dir}/communication_partners_final_results.csv"
    df.to_csv(csv_fname, index=False)
    print(f"Results saved to: {csv_fname}")

def main():
    # Load final communication data from all files
    communication_data = load_final_communication_data()
    
    if not communication_data:
        print("No valid communication data found.")
        return
    
    print(f"Successfully processed {len(communication_data)} files")
    
    # Create the heatmap
    create_communication_heatmap(communication_data)

if __name__ == "__main__":
    main()