import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from itertools import product

def bitstring_to_color(bits):
    """Map a 4 or 2-bit array to a unique RGB color, similar bitstrings = similar colors."""
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
        raise ValueError("Unsupported bitstring length: {}".format(len(bits)))
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

def plot_lattice_and_heatmap(lattice, step, outdir):
    L, _, B = lattice.shape
    rgb_img = np.zeros((L, L, 3))
    for i in range(L):
        for j in range(L):
            rgb_img[i, j] = bitstring_to_color(lattice[i, j])

    # Flatten agents and get their languages
    flat = lattice.reshape(-1, B)
    agent_langs = [tuple(bits) for bits in flat]
    # Get all unique languages present
    unique_langs = sorted(set(agent_langs), key=lambda b: (sum(b), b))
    # Or, to show all possible languages, use:
    # unique_langs = [tuple(bits) for bits in all_bitstrings_sorted(B)]

    # Map language to color
    lang_to_color = {lang: bitstring_to_color(np.array(lang)) for lang in unique_langs}

    # Sort agents by language (block structure)
    agent_indices_sorted = sorted(range(len(agent_langs)), key=lambda idx: (unique_langs.index(agent_langs[idx]), agent_langs[idx]))
    sorted_flat = flat[agent_indices_sorted]

    # Build color heatmap: each bit is colored with the language color if 1, else white
    heatmap = np.ones((len(sorted_flat), B, 3))
    for i, bits in enumerate(sorted_flat):
        color = bitstring_to_color(bits)
        for j, bit in enumerate(bits):
            if bit == 1:
                heatmap[i, j] = color
            else:
                heatmap[i, j] = [1, 1, 1]  # white for 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Heatmap of languages (sorted)
    axes[0].imshow(heatmap, aspect='auto', interpolation='nearest')
    axes[0].set_title("Languages Heatmap (sorted)")
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
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            step, lattice = parse_lattice_line(line, L, B)
            plot_lattice_and_heatmap(lattice, step, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=256)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--r", type=float, default=2)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--K", type=int, default=3)
    args = parser.parse_args()
    process_lattice_timeseries(args.L, args.B, args.gamma, args.alpha, args.r, args.mu, args.K)