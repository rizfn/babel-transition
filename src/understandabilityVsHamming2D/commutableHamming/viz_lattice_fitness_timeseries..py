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

def communicability(a, b):
    """Calculate communicability between two bitstrings."""
    return np.sum(a & b)

def mean_field_distance(language, mean_field):
    """Calculate mean-field distance."""
    return np.mean(np.abs(language.astype(float) - mean_field))

def calculate_fitness(lattice, gamma, alpha):
    """Calculate fitness for each agent in the lattice."""
    L, _, B = lattice.shape
    
    # Calculate mean-field bitstring
    mean_field = np.mean(lattice, axis=(0, 1))
    
    # Initialize fitness array
    fitness = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            agent_lang = lattice[i, j]
            
            # Global fitness: mean-field distance
            global_fitness = gamma * mean_field_distance(agent_lang, mean_field)
            
            # Local fitness: communicability with neighbors
            local_fitness = 0.0
            neighbors = [
                ((i + 1) % L, j),  # down
                ((i - 1) % L, j),  # up
                (i, (j + 1) % L),  # right
                (i, (j - 1) % L)   # left
            ]
            
            for ni, nj in neighbors:
                neighbor_lang = lattice[ni, nj]
                comm = communicability(agent_lang, neighbor_lang)
                local_fitness += (alpha / 4.0) * (comm / B)
            
            fitness[i, j] = global_fitness + local_fitness
    
    return fitness

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

def plot_lattice_and_fitness(lattice, step, gamma, alpha, outdir, color_map=None):
    """Plot lattice colored by language and fitness heatmap side by side."""
    L, _, B = lattice.shape
    
    # Calculate fitness
    fitness = calculate_fitness(lattice, gamma, alpha)
    
    # Create RGB image for lattice
    rgb_img = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            rgb_img[i, j] = bitstring_to_color(lattice[i, j], color_map)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Lattice colored by language
    axes[0].imshow(rgb_img, interpolation='nearest')
    axes[0].set_title("Lattice colored by language")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Right: Fitness heatmap
    im = axes[1].imshow(fitness, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Fitness heatmap")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    
    # Add colorbar for fitness
    plt.colorbar(im, ax=axes[1], label='Fitness')

    # Add statistics
    unique_langs = len(set(tuple(lattice[i, j]) for i in range(L) for j in range(L)))
    fig.suptitle(f"Step {step} | {unique_langs} unique languages | "
                f"Fitness: min={fitness.min():.3f}, max={fitness.max():.3f}, mean={fitness.mean():.3f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, f"frame_{step:04d}.png")
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()

def process_lattice_timeseries(L, B, gamma, alpha, mu, K):
    filename = f"src/understandabilityVsHamming2D/commutableHamming/outputs/latticeTimeseries/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}_K_{K}.tsv"
    outdir = f"src/understandabilityVsHamming2D/commutableHamming/plots/latticeFitnessAnim/frames/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}_K_{K}"
    
    # Create fixed color map once for consistency across all frames
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            if step % 100 != 0:
                continue
            plot_lattice_and_fitness(lattice, step, gamma, alpha, outdir, color_map)

def process_lattice_timeseries_start1(L, B, gamma, alpha, mu, K):
    filename = f"src/understandabilityVsHamming2D/commutableHamming/outputs/latticeTimeseries/start1/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}_K_{K}.tsv"
    outdir = f"src/understandabilityVsHamming2D/commutableHamming/plots/latticeFitnessAnim/start1/frames/L_{L}_g_{gamma}_a_{alpha}_B_{B}_mu_{mu}_K_{K}"
    
    # Create fixed color map once for consistency across all frames
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            plot_lattice_and_fitness(lattice, step, gamma, alpha, outdir, color_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=1)
    args = parser.parse_args()

    # process_lattice_timeseries(args.L, args.B, args.gamma, args.alpha, args.mu, args.K)
    process_lattice_timeseries_start1(args.L, args.B, args.gamma, args.alpha, args.mu, args.K)