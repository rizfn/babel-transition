import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import product
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu from filename (no K parameter)."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+)', filename)  # Fixed regex pattern
    if gamma and alpha and L and B and mu:
        return (float(gamma.group(1)), float(alpha.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)))
    return (None,)*5

def create_bitstring_color_map(B, seed=42):
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

def load_final_lattices(category):
    """Load the final lattice from each file based on category."""
    # Map categories to their subdirectories
    category_dirs = {
        'globalK': 'globalK',
        'globalAlpha': 'globalAlpha', 
        'globalAlphaK': 'globalAlphaK'
    }
    
    if category not in category_dirs:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(category_dirs.keys())}")
    
    subdir = category_dirs[category]
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/{subdir}/L_*_g_*_a_*_B_*_mu_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files for category: {category}")
    
    lattice_data = {}
    
    for filename in tqdm(files, desc="Loading final lattices"):
        gamma, alpha, L, B, mu = extract_params_from_filename(filename)
        
        if gamma is None or alpha is None:
            continue
            
        try:
            last_line = get_last_line_from_file(filename)
            if last_line is None:
                continue
                
            step, lattice = parse_lattice_line(last_line)
            if lattice is not None:
                lattice_data[(gamma, alpha)] = {
                    'lattice': lattice,
                    'step': step,
                    'L': L,
                    'B': B,
                    'mu': mu
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return lattice_data

def get_category_title(category):
    """Get the appropriate title for each category."""
    titles = {
        'globalK': 'Global Competition (K)',
        'globalAlpha': 'Global Alpha Interaction', 
        'globalAlphaK': 'Global Alpha & Competition (K)'
    }
    return titles.get(category, category)

def create_lattice_grid_plot(lattice_data, category):
    """Create a grid plot of lattices organized by gamma and alpha."""
    if not lattice_data:
        print("No lattice data to plot")
        return
    
    # Get unique gamma and alpha values
    gammas = sorted(set(key[0] for key in lattice_data.keys()))
    alphas = sorted(set(key[1] for key in lattice_data.keys()), reverse=True)  # Reverse alpha order so -1 is at bottom
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Get L, B, mu values (assuming all files have same values)
    sample_data = next(iter(lattice_data.values()))
    L = sample_data['L']
    B = sample_data['B']
    mu = sample_data['mu']
    
    # Create color map for bitstrings
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    # Create the figure
    fig, axes = plt.subplots(len(alphas), len(gammas), 
                            figsize=(3*len(gammas), 3*len(alphas)))
    
    # Handle case where we only have one row or column
    if len(alphas) == 1 and len(gammas) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(gammas) == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each lattice
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            ax = axes[i][j]
            
            if (gamma, alpha) in lattice_data:
                data = lattice_data[(gamma, alpha)]
                lattice = data['lattice']
                step = data['step']
                
                # Create RGB image
                rgb_img = np.zeros((L, L, 3))
                for x in range(L):
                    for y in range(L):
                        rgb_img[x, y] = bitstring_to_color(lattice[x, y], color_map)
                
                ax.imshow(rgb_img, interpolation='nearest')
                ax.set_title(f'γ={gamma}, α={alpha}\nStep {step}', fontsize=10)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'γ={gamma}, α={alpha}', fontsize=10)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add overall labels
    category_title = get_category_title(category)
    fig.suptitle(f'Final Lattice States - {category_title}\n(L={L}, B={B}, μ={mu})', fontsize=16)
    fig.text(0.5, 0.02, 'Gamma (Global Interaction Strength)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # Save the plot
    output_dir = f"src/understandabilityVsHamming2D/globalAlphaOrK/plots/{category}/latticeGrid"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/final_lattices_grid_L_{L}_B_{B}_mu_{mu}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {fname}")

def main(category):
    """Main function that takes category as a parameter."""
    # Validate category
    valid_categories = ['globalK', 'globalAlpha', 'globalAlphaK']
    if category not in valid_categories:
        raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
    
    # Load final lattices from all files
    lattice_data = load_final_lattices(category)
    
    if not lattice_data:
        print("No valid lattice data found.")
        return
    
    print(f"Successfully loaded {len(lattice_data)} lattices")
    
    # Create the grid plot
    create_lattice_grid_plot(lattice_data, category)

if __name__ == "__main__":
    # Default behavior - you can change this or call main() directly
    main('globalAlpha')