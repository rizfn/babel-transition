import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from pathlib import Path
from itertools import product

def create_bitstring_color_map(B, seed=31):
    """Create a fixed mapping from bitstrings to shuffled colors from rainbow colormap."""
    # Set random seed for reproducible color assignment
    np.random.seed(seed)
    
    # Get all possible bitstrings
    all_bitstrings = [tuple(bits) for bits in product([0, 1], repeat=B)]
    num_bitstrings = len(all_bitstrings)
    
    # Get colors from rainbow colormap
    cmap = plt.cm.rainbow
    colors = [cmap(i / (num_bitstrings - 1))[:3] for i in range(num_bitstrings)]  # Take only RGB, drop alpha
    
    # Shuffle colors to avoid similar bitstrings getting similar colors
    np.random.shuffle(colors)
    
    # Create mapping
    color_map = dict(zip(all_bitstrings, colors))
    
    # Reset random seed so it doesn't affect other random operations
    np.random.seed(None)
    
    return color_map

def bitstring_to_color(bits, color_map=None):
    """Map a bitstring to a unique RGB color."""
    if len(bits) == 4:
        r = 0.3 + 0.7 * bits[0]
        g = 0.3 + 0.7 * bits[1]
        b = 0.3 + 0.7 * bits[2]
        brightness = 0.3 + 0.5 * bits[3]
        color = np.array([r, g, b]) * brightness
        color = np.clip(color, 0, 1)
    elif len(bits) == 2:
        base = 0.2
        delta = 0.6
        r = base + delta * bits[0]
        g = base + delta * bits[1]
        b = base + delta * (bits[0] ^ bits[1])  # XOR for more separation
        return [r, g, b]
    else:
        # For longer bitstrings, use the provided color map
        if color_map is None:
            raise ValueError("color_map must be provided for bitstrings longer than 4 bits")
        
        bitstring_tuple = tuple(bits)
        if bitstring_tuple not in color_map:
            raise ValueError(f"Bitstring {bitstring_tuple} not found in color_map")
        
        return color_map[bitstring_tuple]
    
    return color

def int_to_bitstring(value, B):
    """Convert integer back to bitstring tuple."""
    bitstring = []
    for i in range(B):
        bitstring.append((value >> i) & 1)
    return tuple(bitstring)

def integer_to_color(value, B, color_map=None):
    """Map an integer directly to a unique RGB color by converting to bitstring first."""
    bits = int_to_bitstring(value, B)
    return bitstring_to_color(bits, color_map)

def load_lattice_data(filepath):
    """Load lattice timeseries data from TSV file."""
    try:
        # Read the data
        data = pd.read_csv(filepath, sep='\t', header=None)
        
        # First column is time step, second column contains lattice states
        time_steps = data.iloc[:, 0].values
        lattice_strings = data.iloc[:, 1].values
        
        # Extract B from the filepath
        path_str = str(filepath)
        b_pos = path_str.find('_B_')
        if b_pos == -1:
            raise ValueError("Could not extract B from filepath")
        
        b_start = b_pos + 3
        b_end = path_str.find('/', b_start)
        if b_end == -1:
            b_end = path_str.find('\\', b_start)  # Windows path separator
        if b_end == -1:
            raise ValueError("Could not extract B from filepath")
        
        B = int(path_str[b_start:b_end])
        
        lattice_matrix = []
        
        for lattice_str in lattice_strings:
            # Split by comma to get individual agent values
            agents = lattice_str.split(',')
            
            # Store each agent's value as integer
            lattice_row = []
            for agent_val in agents:
                # Check if this is integer format or old bitstring format
                if agent_val.isdigit() or (agent_val.startswith('-') and agent_val[1:].isdigit()):
                    # New integer format - keep as integer
                    lattice_row.append(int(agent_val))
                else:
                    # Old bitstring format - convert to integer for consistency
                    bitstring_val = 0
                    for i, bit in enumerate(agent_val):
                        bitstring_val |= (int(bit) << i)
                    lattice_row.append(bitstring_val)
            
            lattice_matrix.append(lattice_row)
        
        L = len(agents) # Number of agents
        
        return time_steps, lattice_matrix, L, B
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None, None, None, None

def parse_filename(filename):
    """Parse parameters from filename: g_X_a_Y_mu_Z.tsv"""
    pattern = r'g_([\d.e-]+)_a_([\d.e-]+)_mu_([\d.e-]+)\.tsv'
    match = re.match(pattern, filename)
    
    if match:
        gamma = float(match.group(1))
        alpha = float(match.group(2))
        mu = float(match.group(3))
        return gamma, alpha, mu
    else:
        raise ValueError(f"Could not parse filename: {filename}")

def plot_all_loaded_lattices(L=None, B=None, seed=31):
    """Plot lattice timeseries for all files in the specified L_X_B_Y directory."""
    
    # Handle seed as single value or array
    if isinstance(seed, (list, tuple, np.ndarray)):
        seeds = seed
    else:
        seeds = [seed]
    
    # Run for each seed
    for current_seed in seeds:
        print(f"Processing with seed: {current_seed}")
        _plot_single_seed(L, B, current_seed)

def _plot_single_seed(L, B, seed):
    """Plot lattice timeseries for a single seed value."""
    # Get script directory
    script_dir = Path(__file__).parent
    
    if L is None or B is None:
        # If no L and B specified, show available directories
        lattice_dir = script_dir / "outputs" / "latticeTimeseriesLoaded"
        if lattice_dir.exists():
            subdirs = [d.name for d in lattice_dir.iterdir() if d.is_dir()]
            print(f"Available directories: {subdirs}")
            print("Please specify L and B parameters, e.g., plot_all_loaded_lattices(L=1024, B=16)")
        else:
            print("latticeTimeseriesLoaded directory not found")
        return
    
    # Find all TSV files in the specific L_B directory
    target_dir = script_dir / "outputs" / "latticeTimeseriesLoaded" / f"L_{L}_B_{B}"
    pattern = str(target_dir / "*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found in {target_dir}")
        return
    
    print(f"Found {len(files)} files in L_{L}_B_{B} directory")
    
    # Specific mutation rates to plot
    target_mu_values = [1e-7, 1e-5, 0.0001, 0.001]
    
    # Parse parameters from filenames and filter for target mutation rates
    file_params = []
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            gamma, alpha, mu = parse_filename(filename)
            if mu in target_mu_values:
                file_params.append((gamma, alpha, mu, filepath))
        except ValueError as e:
            print(f"Skipping file {filename}: {e}")
            continue
    
    # Sort by gamma, then alpha, then mu
    file_params.sort(key=lambda x: (x[0], x[1], x[2]))
    
    n_files = len(file_params)
    if n_files == 0:
        print("No valid files found")
        return
    
    # Arrange all plots in one row
    n_cols = n_files
    n_rows = 1
    
    print(f"Creating {n_rows}x{n_cols} grid for {n_files} files")
    
    # Create color map for bitstrings with the specified seed
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B, seed)
    
    # Create the plot with transparent background
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4))
    fig.patch.set_facecolor('none')  # Transparent figure background
    
    # Handle case where there's only one subplot
    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each file
    for idx, (gamma, alpha, mu, filepath) in enumerate(file_params):
        print(f"Processing file {idx+1}/{n_files}: {os.path.basename(filepath)}")
        
        # Load data (now returns integers)
        time_steps, lattice_matrix, file_L, file_B = load_lattice_data(filepath)
        
        if lattice_matrix is None:
            print(f"Skipping {filepath} - failed to load")
            continue
        
        # Verify that file L and B match expected values
        if file_L != L or file_B != B:
            print(f"Warning: File {filepath} has L={file_L}, B={file_B} but expected L={L}, B={B}")
        
        # Convert lattice data to RGB image (integers to colors)
        n_timesteps = len(lattice_matrix)
        rgb_image = np.zeros((n_timesteps, file_L, 3))
        
        for t, lattice_row in enumerate(lattice_matrix):
            for i, integer_val in enumerate(lattice_row):
                color = integer_to_color(integer_val, file_B, color_map)
                rgb_image[t, i] = color
        
        # Plot on the corresponding subplot
        ax = axes[idx]
        ax.imshow(rgb_image, aspect='auto', interpolation='nearest', origin='lower')
        
        # Format mutation rate for title
        if mu == 1e-7:
            mu_title = r'$\mu=10^{-7}$'
        elif mu == 1e-5:
            mu_title = r'$\mu=10^{-5}$'
        elif mu == 0.0001:
            mu_title = r'$\mu=10^{-4}$'
        elif mu == 0.001:
            mu_title = r'$\mu=10^{-3}$'
        else:
            mu_title = f'μ={mu}'
        
        ax.set_title(mu_title, fontsize=32, pad=10)
        
        # Remove all ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Only add ylabel to the leftmost plot (no xlabel)
        if idx == 0:
            ax.set_ylabel('Time', fontsize=32)
        
        # Make subplot background transparent
        ax.patch.set_facecolor('none')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create output directory and save with seed in filename
    output_dir = script_dir / "plots" / "4mutationsLatticeTimeseries" / f"L_{L}_B_{B}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"1D_mutation_rate_scaling.svg"
    
    plt.savefig(output_path, dpi=300, format='svg', bbox_inches='tight', facecolor='none', edgecolor='none', transparent=True)
    plt.close()
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_all_loaded_lattices(512, 16, seed=62)
