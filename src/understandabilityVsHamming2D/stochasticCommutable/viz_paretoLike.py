import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import combinations, product
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
            # Each cell is a bitstring, convert to list of ints
            bits = [int(b) for b in cell]
            lattice_row.append(bits)
        lattice.append(lattice_row)
    
    # Convert to numpy array
    lattice_array = np.array(lattice)
    return step, lattice_array

def hamming_distance(lang1, lang2):
    """Calculate Hamming distance between two languages."""
    return np.sum(np.array(lang1) != np.array(lang2))

def understandability(lang1, lang2):
    """Calculate understandability (communicability) between two languages."""
    # Count overlapping 1s
    return np.sum(np.array(lang1) & np.array(lang2))

def collect_hu_counts_from_file(filename, B):
    """Collect H-U counts matrix from a single file."""
    # Initialize counts matrix: counts[h][u] = number of pairs with hamming=h, understandability=u
    total_counts = np.zeros((B + 1, B + 1), dtype=np.int64)
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Sample a few lines to avoid memory issues
        if len(lines) > 10:
            step_size = max(1, len(lines) // 10)
            sampled_lines = lines[::step_size]
        else:
            sampled_lines = lines
        
        for line in sampled_lines:
            step, lattice = parse_lattice_line(line)
            if lattice is None:
                continue
            
            # Flatten the lattice to get all languages
            all_languages = []
            for i in range(lattice.shape[0]):
                for j in range(lattice.shape[1]):
                    all_languages.append(tuple(lattice[i, j]))
            
            # Find unique languages and their counts
            unique_languages, counts = np.unique(all_languages, return_counts=True, axis=0)
            
            # Create counts matrix for this step
            step_counts = np.zeros((B + 1, B + 1), dtype=np.int64)
            
            # Calculate distances for all pairs
            for i in range(len(unique_languages)):
                for j in range(i, len(unique_languages)):
                    lang1 = unique_languages[i]
                    lang2 = unique_languages[j]
                    
                    h_dist = hamming_distance(lang1, lang2)
                    u_score = understandability(lang1, lang2)
                    
                    # Skip if indices are out of bounds (shouldn't happen with proper B)
                    if h_dist > B or u_score > B:
                        continue
                    
                    if i == j:
                        # Same language: C(count, 2) pairs
                        pair_count = counts[i] * (counts[i] - 1) // 2
                    else:
                        # Different languages: count1 * count2 pairs
                        pair_count = counts[i] * counts[j]
                    
                    step_counts[h_dist, u_score] += pair_count
            
            total_counts += step_counts
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    
    return total_counts

def load_pareto_data_alpha_mu_raster(L, B, gamma, output_dir="rasterscanMu"):
    """Load Pareto data for all alpha-mu combinations from files."""
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/latticeTimeseries/{output_dir}/L_{L}_g_{gamma}_a_*_B_{B}_mu_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files in {output_dir} folder matching L={L}, B={B}, gamma={gamma}")
    
    pareto_data = {}
    
    for filename in tqdm(files, desc="Loading Pareto data"):
        gamma_file, alpha, L_file, B_file, mu = extract_params_from_filename(filename)
        
        if gamma_file is None or alpha is None or mu is None:
            continue
            
        # Double-check that the extracted parameters match what we're looking for
        if L_file != L or B_file != B or gamma_file != gamma:
            continue
        
        # Collect counts matrix from this file
        counts_matrix = collect_hu_counts_from_file(filename, B)
        
        if np.sum(counts_matrix) > 0:
            pareto_data[(alpha, mu)] = {
                'counts_matrix': counts_matrix,
                'L': L_file,
                'B': B_file,
                'gamma': gamma_file,
                'alpha': alpha,
                'mu': mu
            }
    
    return pareto_data


def plot_single_pareto_subplot_from_counts(ax, counts_matrix, B, alpha, mu, min_count_per_point=1, show_legend=False):
    """Plot a single Pareto-like plot from counts matrix."""
    # Apply minimum count threshold
    filtered_counts = (counts_matrix // min_count_per_point).astype(int)
    
    # Convert counts matrix to coordinate arrays with jitter
    hamming_distances = []
    understandabilities = []
    
    for h in range(B + 1):
        for u in range(B + 1):
            count = filtered_counts[h, u]
            if count > 0:
                # Add jittered points based on count (sample to avoid too many points)
                sample_size = min(count, 1000)  # Limit points for visualization
                if sample_size > 0:
                    # Add radial jitter
                    jitter_amount = 0.15
                    angles = np.random.uniform(0, 2 * np.pi, sample_size)
                    radii = np.random.uniform(0, jitter_amount, sample_size)
                    
                    h_jittered = h + radii * np.cos(angles)
                    u_jittered = u + radii * np.sin(angles)
                    
                    hamming_distances.extend(h_jittered)
                    understandabilities.extend(u_jittered)
    
    # Create scatter plot
    if hamming_distances:
        ax.scatter(hamming_distances, understandabilities, alpha=0.4, s=3, color='blue', 
                   edgecolors='none')
    
    # Plot theoretical boundary
    h_boundary = np.arange(0, B + 1)
    u_boundary = B - h_boundary
    
    ax.plot(h_boundary, u_boundary, 'r--', linewidth=1, alpha=0.7)
    
    # Fill the impossible region
    ax.fill_between(h_boundary, u_boundary, B, alpha=0.15, color='red')
    
    # Customize plot
    ax.set_xlim(-0.5, B + 0.5)
    ax.set_ylim(-0.5, B + 0.5)
    ax.set_title(f'α={alpha}, μ={mu:.6f}', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Remove tick labels for cleaner look in grid
    ax.set_xticks([])
    ax.set_yticks([])

def viz_single_file(L, B, gamma, alpha, mu, output_dir="rasterscanMu", max_files=10, min_count_per_point=1):
    """Create scatter plot of Hamming distance vs understandability for a single parameter set."""
    # Get files matching the parameters
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, f"outputs/latticeTimeseries/{output_dir}/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return
    
    # Limit number of files to process
    if len(files) > max_files:
        files = files[:max_files]
    
    print(f"Processing {len(files)} files...")
    
    # Collect all counts matrices and sum them
    total_counts = np.zeros((B + 1, B + 1), dtype=np.int64)
    for filename in tqdm(files, desc="Processing files"):
        counts_matrix = collect_hu_counts_from_file(filename, B)
        total_counts += counts_matrix
    
    if np.sum(total_counts) == 0:
        print("No language pairs found!")
        return
    
    print(f"Total pairs: {np.sum(total_counts)}")
    
    # Apply minimum count threshold
    filtered_counts = (total_counts // min_count_per_point).astype(int)
    
    # Convert counts matrix to coordinate arrays for plotting
    hamming_distances = []
    understandabilities = []
    
    for h in range(B + 1):
        for u in range(B + 1):
            count = filtered_counts[h, u]
            if count > 0:
                # Add jittered points based on count
                sample_size = min(count, 2000)  # Limit points for visualization
                if sample_size > 0:
                    # Add radial jitter
                    jitter_amount = 0.2
                    np.random.seed(42)  # For reproducible jitter
                    angles = np.random.uniform(0, 2 * np.pi, sample_size)
                    radii = np.random.uniform(0, jitter_amount, sample_size)
                    
                    h_jittered = h + radii * np.cos(angles)
                    u_jittered = u + radii * np.sin(angles)
                    
                    hamming_distances.extend(h_jittered)
                    understandabilities.extend(u_jittered)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    if hamming_distances:
        ax.scatter(hamming_distances, understandabilities, alpha=0.4, s=10, color='blue', 
                   label='Observed pairs', edgecolors='none')
    
    # Plot theoretical boundary
    h_boundary = np.arange(0, B + 1)
    u_boundary = B - h_boundary
    
    ax.plot(h_boundary, u_boundary, 'r--', linewidth=2, alpha=0.5, label='Theoretical boundary')
    
    # Fill the impossible region
    ax.fill_between(h_boundary, u_boundary, B, alpha=0.2, color='red', 
                   label='Impossible region')
    
    # Customize plot
    ax.set_xlabel('Hamming Distance', fontsize=12)
    ax.set_ylabel('Understandability (Communicability)', fontsize=12)
    ax.set_title(f'Hamming Distance vs Understandability\n(L={L}, B={B}, γ={gamma}, α={alpha}, μ={mu}, min_count={min_count_per_point})', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(-0.5, B + 0.5)
    ax.set_ylim(-0.5, B + 0.5)
    
    # Add statistics (using original counts for statistics)
    total_pairs = np.sum(total_counts)
    mean_h = np.sum(np.arange(B + 1)[:, np.newaxis] * total_counts) / total_pairs
    mean_u = np.sum(np.arange(B + 1)[np.newaxis, :] * total_counts) / total_pairs
    
    textstr = f'Total pairs: {total_pairs}\n'
    textstr += f'Mean H: {mean_h:.2f}\n'
    textstr += f'Mean U: {mean_u:.2f}\n'
    textstr += f'Min count/point: {min_count_per_point}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save the plot
    plot_output_dir = os.path.join(base_dir, f"plots/paretoLike/{output_dir}")
    os.makedirs(plot_output_dir, exist_ok=True)
    fname = os.path.join(plot_output_dir, f"L_{L}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}_mincount_{min_count_per_point}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")

def create_pareto_alpha_mu_raster_plot(pareto_data, L, B, gamma, min_count_per_point=1):
    """Create a grid plot of Pareto-like plots organized by mu (x-axis) and alpha (y-axis)."""
    if not pareto_data:
        print("No Pareto data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in pareto_data.keys()))
    mus = sorted(set(key[1] for key in pareto_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create the figure
    alphas_reversed = list(reversed(alphas))
    fig, axes = plt.subplots(len(alphas), len(mus), 
                            figsize=(2.5*len(mus), 2.5*len(alphas)))
    
    # Handle case where we only have one row or column
    if len(alphas) == 1 and len(mus) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(mus) == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each Pareto plot
    for i, alpha in enumerate(alphas_reversed):
        for j, mu in enumerate(mus):
            ax = axes[i][j]
            
            if (alpha, mu) in pareto_data:
                data = pareto_data[(alpha, mu)]
                counts_matrix = data['counts_matrix']
                
                plot_single_pareto_subplot_from_counts(ax, counts_matrix, B, alpha, mu, min_count_per_point)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'α={alpha}, μ={mu:.6f}', fontsize=8)
                ax.set_xlim(0, B)
                ax.set_ylim(0, B)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall labels
    fig.suptitle(f'Hamming vs Understandability Raster: Alpha vs Mu\n(L={L}, B={B}, γ={gamma}, min_count={min_count_per_point})', fontsize=16)
    fig.text(0.5, 0.02, 'Mu (Mutation Rate)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # Save the plot
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "plots/paretoLike/rasterscanMu")
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"alpha_mu_raster_pareto_L_{L}_B_{B}_gamma_{gamma}_mincount_{min_count_per_point}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")

def viz_raster(L, B, gamma, output_dir="rasterscanMu", min_count_per_point=1):
    """Create a raster scan of Pareto-like plots for all alpha-mu combinations."""
    print(f"Creating Pareto raster scan with parameters: L={L}, B={B}, gamma={gamma}")
    
    # Load Pareto data for all alpha-mu combinations
    pareto_data = load_pareto_data_alpha_mu_raster(L, B, gamma, output_dir=output_dir)
    
    if not pareto_data:
        print("No valid Pareto data found.")
        return
    
    print(f"Successfully loaded {len(pareto_data)} parameter combinations")
    
    # Create the grid plot
    create_pareto_alpha_mu_raster_plot(pareto_data, L, B, gamma, min_count_per_point)

if __name__ == "__main__":
    # Parameters
    L = 256
    B = 16
    gamma = 1
    alpha = 1.4
    mu = 0.000464159
    min_count_per_point = 10000
    
    # Create single file plot
    viz_single_file(L=L, B=B, gamma=gamma, alpha=alpha, mu=mu, 
                    output_dir="rasterscanMu", min_count_per_point=min_count_per_point)
    
    # Create raster scan plot
    viz_raster(L=L, B=B, gamma=gamma, output_dir="rasterscanMu", min_count_per_point=min_count_per_point)