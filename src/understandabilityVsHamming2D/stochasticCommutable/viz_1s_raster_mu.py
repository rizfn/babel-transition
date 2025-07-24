import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)(?:_|\.tsv)', filename)
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

def count_ones_in_lattice(lattice):
    """Count the average number of 1s across all bitstrings in the lattice."""
    total_ones = 0
    total_bits = 0
    
    L, _, B = lattice.shape
    for i in range(L):
        for j in range(L):
            bitstring = lattice[i, j]
            ones_count = np.sum(bitstring)
            total_ones += ones_count
            total_bits += B
    
    return total_ones / total_bits if total_bits > 0 else 0

def get_ones_distribution(lattice):
    """Get the distribution of 1s counts in all bitstrings in the lattice."""
    L, _, B = lattice.shape
    ones_counts = []
    
    for i in range(L):
        for j in range(L):
            bitstring = lattice[i, j]
            ones_count = np.sum(bitstring)
            ones_counts.append(ones_count)
    
    return np.array(ones_counts)

def load_lattice_data(L, B, gamma):
    """Load final lattices and compute both average 1s counts and distributions for each alpha-mu combination."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscanMu/L_{L}_B_{B}/g_{gamma}_a_*_mu_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}, {}
    
    print(f"Found {len(files)} files matching L={L}, B={B}, gamma={gamma}")
    
    ones_data = {}
    distribution_data = {}
    
    for filename in tqdm(files, desc="Processing lattice data"):
        gamma_file, alpha, L_file, B_file, mu = extract_params_from_filename(filename)
        
        if gamma_file is None or alpha is None or mu is None:
            continue
            
        # Double-check that the extracted parameters match what we're looking for
        if L_file != L or B_file != B or gamma_file != gamma:
            continue
            
        try:
            last_line = get_last_line_from_file(filename)
            if last_line is None:
                continue
                
            step, lattice = parse_lattice_line(last_line)
            if lattice is not None:
                # Compute average 1s count
                avg_ones = count_ones_in_lattice(lattice)
                ones_data[(alpha, mu)] = {
                    'avg_ones': avg_ones,
                    'step': step,
                    'alpha': alpha,
                    'mu': mu
                }
                
                # Compute 1s distribution
                ones_counts = get_ones_distribution(lattice)
                distribution_data[(alpha, mu)] = {
                    'ones_counts': ones_counts,
                    'step': step,
                    'alpha': alpha,
                    'mu': mu
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return ones_data, distribution_data

def create_ones_heatmap(ones_data, L, B, gamma):
    """Create a heatmap of average number of 1s organized by alpha and mu."""
    if not ones_data:
        print("No data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in ones_data.keys()))
    mus = sorted(set(key[1] for key in ones_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create 2D array for heatmap
    heatmap_data = np.full((len(alphas), len(mus)), np.nan)
    
    for i, alpha in enumerate(alphas):
        for j, mu in enumerate(mus):
            if (alpha, mu) in ones_data:
                heatmap_data[i, j] = ones_data[(alpha, mu)]['avg_ones']
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Flip the heatmap vertically so highest alpha is at top
    heatmap_data_flipped = np.flipud(heatmap_data)
    alphas_flipped = list(reversed(alphas))
    
    im = ax.imshow(heatmap_data_flipped, cmap='viridis', aspect='auto', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(mus)))
    ax.set_yticks(range(len(alphas)))
    ax.set_xticklabels([f'{mu:.4f}' for mu in mus], rotation=45)
    ax.set_yticklabels([f'{alpha:.1f}' for alpha in alphas_flipped])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Number of 1s per Bitstring', rotation=270, labelpad=20)
    
    # Add text annotations with the values
    for i in range(len(alphas)):
        for j in range(len(mus)):
            if not np.isnan(heatmap_data_flipped[i, j]):
                text = f'{heatmap_data_flipped[i, j]:.3f}'
                ax.text(j, i, text, ha="center", va="center", 
                       color="white" if heatmap_data_flipped[i, j] < 0.5 else "black",
                       fontsize=8)
    
    # Labels and title
    ax.set_xlabel('Mu (Mutation Rate)', fontsize=12)
    ax.set_ylabel('Alpha (Local Interaction Strength)', fontsize=12)
    ax.set_title(f'Average Number of 1s per Bitstring\n(L={L}, B={B}, γ={gamma})', fontsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "src/understandabilityVsHamming2D/stochasticCommutable/plots/ones_fraction"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/ones_heatmap_L_{L}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Heatmap saved to: {fname}")

def create_distribution_grid(distribution_data, L, B, gamma):
    """Create a grid plot of 1s distributions organized by mu (x-axis) and alpha (y-axis)."""
    if not distribution_data:
        print("No distribution data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in distribution_data.keys()))
    mus = sorted(set(key[1] for key in distribution_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create the figure
    # Reverse alphas so highest alpha is at top
    alphas_reversed = list(reversed(alphas))
    fig, axes = plt.subplots(len(alphas), len(mus), 
                            figsize=(3*len(mus), 2.5*len(alphas)))
    
    # Handle case where we only have one row or column
    if len(alphas) == 1 and len(mus) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(mus) == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each distribution
    for i, alpha in enumerate(alphas_reversed):  # Highest alpha at top (row 0)
        for j, mu in enumerate(mus):              # Lowest mu at left (column 0)
            ax = axes[i][j]
            
            if (alpha, mu) in distribution_data:
                data = distribution_data[(alpha, mu)]
                ones_counts = data['ones_counts']
                step = data['step']
                
                # Create histogram
                bins = np.arange(0, B+2) - 0.5  # Bins centered on integers
                ax.hist(ones_counts, bins=bins, density=True, alpha=0.7, 
                       color='steelblue', edgecolor='black', linewidth=0.5)
                
                # Statistics
                mean_ones = np.mean(ones_counts)
                std_ones = np.std(ones_counts)
                
                ax.set_title(f'α={alpha}, μ={mu:.4f}\nμ={mean_ones:.2f}, σ={std_ones:.2f}', 
                           fontsize=8)
                ax.set_xlim(-0.5, B+0.5)
                ax.set_xticks(range(0, B+1, max(1, B//4)))
                
                # Only show y-axis labels on leftmost plots
                if j == 0:
                    ax.set_ylabel('Density', fontsize=8)
                else:
                    ax.set_yticklabels([])
                
                # Only show x-axis labels on bottom plots
                if i == len(alphas_reversed) - 1:
                    ax.set_xlabel('Number of 1s', fontsize=8)
                else:
                    ax.set_xticklabels([])
                
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'α={alpha}, μ={mu:.4f}', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall labels
    fig.suptitle(f'Distribution of 1s in Bitstrings: Alpha vs Mu Raster\n(L={L}, B={B}, γ={gamma})', 
                fontsize=16)
    fig.text(0.5, 0.02, 'Mu (Mutation Rate)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', 
             rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # Save the plot
    output_dir = "src/understandabilityVsHamming2D/stochasticCommutable/plots/ones_fraction"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/ones_distribution_grid_L_{L}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Distribution grid saved to: {fname}")


def main(L=256, B=16, gamma=1):
    """Main function that generates both visualizations."""
    print(f"Loading lattice data for parameters: L={L}, B={B}, gamma={gamma}")
    
    # Load data and compute both average 1s counts and distributions
    ones_data, distribution_data = load_lattice_data(L, B, gamma)
    
    if not ones_data and not distribution_data:
        print("No valid data found.")
        return
    
    print(f"Successfully processed {len(ones_data)} parameter combinations")
    
    # Create both visualizations
    if ones_data:
        create_ones_heatmap(ones_data, L, B, gamma)
    
    if distribution_data:
        create_distribution_grid(distribution_data, L, B, gamma)

if __name__ == "__main__":
    main(L=256, B=16, gamma=1)