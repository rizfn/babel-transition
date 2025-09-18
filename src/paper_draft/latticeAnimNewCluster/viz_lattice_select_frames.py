import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from itertools import product

def create_bitstring_color_map(B, seed=40):
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

def create_optimized_bitstring_color_map(unique_bitstrings, seed=40):
    """Create a fixed mapping from unique bitstrings to shuffled colors from rainbow colormap."""
    # Set random seed for reproducible color assignment
    np.random.seed(seed)
    
    num_bitstrings = len(unique_bitstrings)
    
    # Get colors from rainbow colormap
    cmap = plt.cm.rainbow
    colors = [cmap(i / (num_bitstrings - 1))[:3] for i in range(num_bitstrings)]  # Take only RGB, drop alpha
    
    # Shuffle colors to avoid similar bitstrings getting similar colors
    np.random.shuffle(colors)
    
    # Create mapping
    color_map = dict(zip(unique_bitstrings, colors))
    
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

def all_bitstrings_sorted(B):
    """Return all bitstrings of length B, sorted by Hamming weight then lex."""
    bitstrings = [np.array(bits) for bits in product([0,1], repeat=B)]
    bitstrings.sort(key=lambda b: (np.sum(b), b.tolist()))
    return bitstrings

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

def plot_lattice_clean_svg(lattice, step, outdir, color_map=None):
    """Plot clean lattice visualization without titles, axes, or ticks, saved as SVG."""
    L, _, B = lattice.shape
    
    rgb_img = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            rgb_img[i, j] = bitstring_to_color(lattice[i, j], color_map)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor('none')  # Transparent figure background
    
    # Remove all visual elements
    ax.imshow(rgb_img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, f"lattice_step_{step}.svg")
    plt.savefig(outname, dpi=300, format='svg', bbox_inches='tight', pad_inches=0, 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()

def collect_unique_bitstrings(filename, L, B, target_frames):
    """Collect all unique bitstrings from target frames in the file using optimized numpy operations."""
    unique_bitstrings = set()
    
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            
            # Only collect from target frames
            if step in target_frames:
                # Reshape lattice to (L*L, B) and convert each row to tuple
                reshaped_lattice = lattice.reshape(-1, B)
                # Convert to tuples efficiently using map
                bitstring_tuples = map(tuple, reshaped_lattice)
                # Update set with all bitstrings from this frame
                unique_bitstrings.update(bitstring_tuples)
    
    return unique_bitstrings

def process_lattice_timeseries(L, B, gamma, alpha, mu, subdir="long", seed=40):
    """Process lattice timeseries from the specified subdirectory."""
    # Get the current directory (stochasticCommutable)
    base_dir = os.path.dirname(__file__)
    filename = os.path.join(base_dir, f"outputs/latticeTimeseries/{subdir}/L_{L}_B_{B}/g_{gamma}_a_{alpha}_mu_{mu}.tsv")
    outdir = os.path.join(base_dir, f"plots/latticeAnim/svg_frames/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}")
    
    # Specific frames to visualize
    target_frames = {28700, 28900, 29100, 30000}
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    print(f"Processing file: {filename}")
    print(f"Output directory: {outdir}")
    print(f"Target frames: {sorted(target_frames)}")
    
    # First pass: collect all unique bitstrings from target frames
    print("Collecting unique bitstrings...")
    unique_bitstrings = collect_unique_bitstrings(filename, L, B, target_frames)
    print(f"Found {len(unique_bitstrings)} unique bitstrings")
    
    # Create optimized color map for only the bitstrings that are actually used
    color_map = None
    if B > 4:
        color_map = create_optimized_bitstring_color_map(unique_bitstrings, seed)
    elif B <= 4:
        # For small B, still collect unique bitstrings for consistency
        unique_bitstrings = collect_unique_bitstrings(filename, L, B, target_frames)
        print(f"Found {len(unique_bitstrings)} unique bitstrings")
    
    # Second pass: generate visualizations
    print("Generating visualizations...")
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            
            # Only process target frames
            if step in target_frames:
                print(f"Processing frame {step}")
                plot_lattice_clean_svg(lattice, step, outdir, color_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=0.8)
    parser.add_argument("--mu", type=float, default=0.0001)
    parser.add_argument("--subdir", type=str, default="long", help="Subdirectory (e.g., 'long', 'rasterscanMu')")
    parser.add_argument("--seed", type=int, default=52, help="Random seed for colormap generation")
    args = parser.parse_args()

    process_lattice_timeseries(args.L, args.B, args.gamma, args.alpha, args.mu, args.subdir, args.seed)