import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from itertools import product

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

def plot_lattice_and_heatmap(lattice, step, outdir, color_map=None):
    L, _, B = lattice.shape
    
    rgb_img = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            rgb_img[i, j] = bitstring_to_color(lattice[i, j], color_map)

    # Flatten agents and get their languages
    flat = lattice.reshape(-1, B)
    agent_langs = [tuple(bits) for bits in flat]
    
    # Get all unique languages present for sorting reference
    unique_langs = sorted(set(agent_langs), key=lambda b: (sum(b), b))

    # Sort agents by language (block structure)
    agent_indices_sorted = sorted(range(len(agent_langs)), key=lambda idx: (unique_langs.index(agent_langs[idx]), agent_langs[idx]))
    sorted_flat = flat[agent_indices_sorted]

    # Build color heatmap: each bit is colored with the language color if 1, else white
    heatmap = np.ones((len(sorted_flat), B, 3))
    for i, bits in enumerate(sorted_flat):
        color = bitstring_to_color(bits, color_map)
        for j, bit in enumerate(bits):
            if bit == 1:
                heatmap[i, j] = color
            else:
                heatmap[i, j] = [1, 1, 1]  # white for 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Heatmap of languages (sorted by language, showing all agents)
    axes[0].imshow(heatmap, aspect='auto', interpolation='nearest')
    axes[0].set_title(f"Languages Heatmap (sorted) - {len(unique_langs)} unique")
    axes[0].set_xlabel("Bit Position")
    axes[0].set_ylabel("Agent (sorted by language)")

    # Lattice colored by bitstring
    axes[1].imshow(rgb_img, interpolation='nearest')
    axes[1].set_title("Lattice colored by bitstring")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    plt.suptitle(f"Step {step}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, f"frame_{step:04d}.png")
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()

def process_lattice_timeseries(L, B, gamma, alpha, r, mu, K):
    filename = f"src/understandabilityVsHammingSmall2D/outputs/latticeNbrVsGlobalTimeseries/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}.tsv"
    outdir = f"src/understandabilityVsHammingSmall2D/plots/latticeAnimNbrVsGlobal/frames/L_{L}_g_{gamma}_a_{alpha}_r_{r}_B_{B}_mu_{mu}_K_{K}"
    
    # Create fixed color map once for consistency across all frames
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            plot_lattice_and_heatmap(lattice, step, outdir, color_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--r", type=float, default=1)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=3)
    args = parser.parse_args()
    process_lattice_timeseries(args.L, args.B, args.gamma, args.alpha, args.r, args.mu, args.K)