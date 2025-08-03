import os
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    if gamma and alpha and mu:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)))
    return (None,)*3

def parse_lattice_line(line):
    """Parse a single line into a 2D lattice array."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    step = int(parts[0])
    lattice_data = parts[1]
    rows = lattice_data.split(';')
    lattice = []
    for row in rows:
        cells = row.split(',')
        lattice_row = []
        for cell in cells:
            bits = tuple(int(b) for b in cell)
            lattice_row.append(bits)
        lattice.append(lattice_row)
    return step, lattice

def bitstring_to_int(bitstring):
    """Convert a bitstring tuple to an integer for easier processing."""
    return int(''.join(str(b) for b in bitstring), 2)

def mean_field_distance(language, mean_field):
    """Calculate mean-field distance."""
    return np.mean(np.abs(np.array(language, dtype=float) - mean_field))

def communicability(a, b):
    """Calculate communicability between two bitstrings."""
    return np.sum(np.array(a) & np.array(b))

def calculate_gamma_fitness(language, mean_field, gamma):
    """Calculate the gamma (global) fitness term."""
    return gamma * mean_field_distance(language, mean_field)

def calculate_total_fitness_isolated(language, mean_field, gamma, alpha, B):
    """Calculate total fitness for an agent surrounded by identical neighbors."""
    # Gamma term
    gamma_fitness = calculate_gamma_fitness(language, mean_field, gamma)
    
    # Local term: when all neighbors are identical, communicability is sum of 1s
    ones_count = np.sum(language)
    local_fitness = alpha * (ones_count / B)
    
    return gamma_fitness + local_fitness

def find_file_for_params(L, B, gamma, alpha, mu, subdir="rasterscanMu"):
    """Find the file corresponding to given parameters in the specified subdir."""
    pattern = os.path.join(
        os.path.dirname(__file__),
        f"outputs/latticeTimeseries/{subdir}/L_{L}_B_{B}/g_{gamma}_a_{alpha}_mu_{mu}.tsv"
    )
    if os.path.exists(pattern):
        return pattern
    # Try to find file with scientific notation for mu
    folder = os.path.join(
        os.path.dirname(__file__),
        f"outputs/latticeTimeseries/{subdir}/L_{L}_B_{B}"
    )
    if not os.path.exists(folder):
        return None
    for fname in os.listdir(folder):
        if re.fullmatch(rf"g_{gamma}_a_{alpha}_mu_.*\.tsv", fname):
            mu_match = re.search(r'mu_([0-9eE\.\+-]+)', fname)
            if mu_match:
                mu_in_file = float(mu_match.group(1))
                if f"mu_{mu}" in fname or np.isclose(float(mu), mu_in_file, rtol=1e-8):
                    return os.path.join(folder, fname)
    return None

def analyze_clusters_at_timestep(lattice, gamma, alpha, B):
    """Analyze clusters in the lattice and return cluster statistics."""
    L = len(lattice)
    
    # Convert lattice to integer representation for easier processing
    int_lattice = np.array([[bitstring_to_int(cell) for cell in row] for row in lattice])
    
    # Calculate mean field
    flat_languages = [cell for row in lattice for cell in row]
    mean_field = np.mean(flat_languages, axis=0)
    
    # Count populations for each language
    flat_int = int_lattice.flatten()
    populations = Counter(flat_int)
    
    cluster_data = []
    
    for lang_int, population in populations.items():
        # Convert back to bitstring
        language = tuple(int(b) for b in format(lang_int, f'0{B}b'))
        
        # Calculate gamma fitness
        gamma_fitness = calculate_gamma_fitness(language, mean_field, gamma)
        
        # Calculate total fitness for isolated cluster
        total_fitness = calculate_total_fitness_isolated(language, mean_field, gamma, alpha, B)
        
        cluster_data.append({
            'language': language,
            'population': population,
            'gamma_fitness': gamma_fitness,
            'total_fitness': total_fitness
        })
    
    return cluster_data

def load_and_analyze_timeseries(filename, gamma, alpha, B, n_points=10):
    """Load file and analyze n_points evenly spaced timesteps."""
    # First pass: collect all timesteps (just read timestep, don't parse lattice yet)
    timesteps = []
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                timesteps.append(int(parts[0]))
    
    if len(timesteps) < n_points:
        print(f"Warning: Only {len(timesteps)} timesteps available, using all")
        selected_indices = list(range(len(timesteps)))
    else:
        # Select evenly spaced indices
        selected_indices = np.linspace(0, len(timesteps)-1, n_points, dtype=int)
    
    # Get the timesteps we want to analyze
    target_timesteps = set(timesteps[i] for i in selected_indices)
    
    all_cluster_data = []
    
    print(f"Analyzing {len(selected_indices)} timesteps...")
    
    # Second pass: only parse lattices for the timesteps we need
    with open(filename, "r") as f:
        for line in tqdm(f, desc="Processing file"):
            line = line.strip()
            if not line:
                continue
            
            # Quick check of timestep before expensive parsing
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            step = int(parts[0])
            if step not in target_timesteps:
                continue
            
            # Only parse lattice if we need this timestep
            step, lattice = parse_lattice_line(line)
            if lattice is None:
                continue
                
            cluster_data = analyze_clusters_at_timestep(lattice, gamma, alpha, B)
            
            # Add timestep info to each cluster
            for cluster in cluster_data:
                cluster['timestep'] = step
            
            all_cluster_data.extend(cluster_data)
    
    return all_cluster_data

def plot_cluster_fitness_analysis(L, B, gamma, alpha, mu, subdir="long", n_points=10):
    """Create scatterplots of cluster population vs fitness with population histogram."""
    filename = find_file_for_params(L, B, gamma, alpha, mu, subdir=subdir)
    if filename is None:
        print(f"No file found for L={L}, B={B}, gamma={gamma}, alpha={alpha}, mu={mu} in {subdir}")
        return

    print(f"Found file: {filename}")
    
    # Load and analyze data
    cluster_data = load_and_analyze_timeseries(filename, gamma, alpha, B, n_points)
    
    if not cluster_data:
        print("No cluster data found")
        return
    
    # Extract data for plotting
    populations = [cluster['population'] for cluster in cluster_data]
    gamma_fitness = [cluster['gamma_fitness'] for cluster in cluster_data]
    total_fitness = [cluster['total_fitness'] for cluster in cluster_data]
    
    # Create the plot with five subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Create gridspec for custom layout: 2 rows, 3 columns
    # Top row: scatter plots and population histogram
    # Bottom row: fitness histograms (spanning 2 columns each)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1, 0.4], 
                         hspace=0.3, wspace=0)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Gamma fitness scatter
    ax2 = fig.add_subplot(gs[0, 1])  # Total fitness scatter
    ax3 = fig.add_subplot(gs[0, 2])  # Population histogram
    ax4 = fig.add_subplot(gs[1, 0])  # Gamma fitness histogram
    ax5 = fig.add_subplot(gs[1, 1])  # Total fitness histogram
    
    # Plot 1: Gamma Fitness vs Population
    ax1.scatter(gamma_fitness, populations, alpha=0.4, s=30)
    ax1.set_xlabel('Gamma Fitness')
    ax1.set_ylabel('Population Size')
    ax1.set_title('Gamma Fitness vs Cluster Population')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Fitness vs Population
    ax2.scatter(total_fitness, populations, alpha=0.4, s=30)
    ax2.set_xlabel('Total Fitness (Isolated)')
    ax2.set_ylabel('Population Size')
    ax2.set_title('Total Fitness vs Cluster Population')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Population histogram (rotated to align with y-axis of scatter plots)
    counts, bins, patches = ax3.hist(populations, bins=30, orientation='horizontal', 
                                   alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Find the second highest bin count to set appropriate x-axis scale
    # This avoids the issue where the first bin (small populations) dominates
    sorted_counts = np.sort(counts)
    if len(sorted_counts) >= 2:
        # Use second highest count as reference
        second_highest = sorted_counts[-2]
        ax3.set_xlim(0, second_highest * 1.2)  # Add 20% padding
    elif len(sorted_counts) >= 1:
        # Fallback to highest if only one bin
        ax3.set_xlim(0, sorted_counts[-1] * 1.2)
    
    ax3.set_xlabel('Count')
    ax3.set_ylabel('Population Size')
    ax3.set_title('Population\nDistribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gamma Fitness histogram
    ax4.hist(gamma_fitness, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Gamma Fitness')
    ax4.set_ylabel('Count')
    ax4.set_title('Gamma Fitness Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Total Fitness histogram
    ax5.hist(total_fitness, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Total Fitness (Isolated)')
    ax5.set_ylabel('Count')
    ax5.set_title('Total Fitness Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Align y-axis of population histogram with scatter plots
    y_min = min(min(populations), ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(max(populations), ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax3.set_ylim(y_min, y_max)
    
    # Remove y-axis labels from middle plot to avoid clutter
    ax2.set_ylabel('')
    ax2.tick_params(left=False, labelleft=False)
    
    # Overall title
    fig.suptitle(f'Cluster Fitness Analysis\nL={L}, B={B}, γ={gamma}, α={alpha}, μ={mu}', 
                 fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save plot
    output_dir = f"src/understandabilityVsHamming2D/stochasticCommutable/plots/populationCorrelations/{subdir}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/cluster_fitness_analysis_L_{L}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fname}")
    
    # Print some statistics
    print(f"\nAnalyzed {len(cluster_data)} clusters across {n_points} timesteps")
    print(f"Population range: {min(populations)} - {max(populations)}")
    print(f"Gamma fitness range: {min(gamma_fitness):.4f} - {max(gamma_fitness):.4f}")
    print(f"Total fitness range: {min(total_fitness):.4f} - {max(total_fitness):.4f}")
    
    # Population distribution statistics
    unique_pops, counts = np.unique(populations, return_counts=True)
    print(f"Most common population sizes:")
    for pop, count in sorted(zip(unique_pops, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Population {pop}: {count} clusters")

def main():
    L = 256
    B = 16
    gamma = 1
    alpha = 1.2
    mu = 0.0001
    subdir = "rasterscanMu"
    n_points = 100
    
    plot_cluster_fitness_analysis(L, B, gamma, alpha, mu, subdir=subdir, n_points=n_points)

if __name__ == "__main__":
    main()