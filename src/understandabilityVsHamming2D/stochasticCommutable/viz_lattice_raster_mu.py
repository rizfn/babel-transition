import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import product
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    if gamma and alpha and mu:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)))
    return (None,)*3

def create_bitstring_color_map(B, seed=42):
    """Create a fixed mapping from bitstrings to shuffled colors from rainbow colormap."""
    np.random.seed(seed)
    all_bitstrings = [tuple(bits) for bits in product([0, 1], repeat=B)]
    num_bitstrings = len(all_bitstrings)
    cmap = plt.cm.rainbow
    colors = [cmap(i / (num_bitstrings - 1))[:3] for i in range(num_bitstrings)]
    np.random.shuffle(colors)
    color_map = dict(zip(all_bitstrings, colors))
    np.random.seed(None)
    return color_map

def get_rainbow_color_list(n_colors=65536, seed=42):
    """Get a shuffled list of colors from the rainbow colormap."""
    np.random.seed(seed)
    cmap = plt.cm.rainbow
    colors = [cmap(i / (n_colors - 1))[:3] for i in range(n_colors)]
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors

def bitstring_to_color(bits, color_map=None, rainbow_colors=None):
    """Map a bitstring to a unique RGB color."""
    if len(bits) == 4:
        r = 0.3 + 0.7 * bits[0]
        g = 0.3 + 0.7 * bits[1]
        b = 0.3 + 0.7 * bits[2]
        brightness = 0.3 + 0.5 * bits[3]
        color = np.array([r, g, b]) * brightness
        color = np.clip(color, 0, 1)
        return color
    elif len(bits) == 2:
        base = 0.2
        delta = 0.6
        r = base + delta * bits[0]
        g = base + delta * bits[1]
        b = base + delta * (bits[0] ^ bits[1])
        return [r, g, b]
    elif color_map is not None:
        bitstring_tuple = tuple(bits)
        if bitstring_tuple in color_map:
            return color_map[bitstring_tuple]
        # fallback if not found
        if rainbow_colors is not None:
            idx = int(''.join(str(b) for b in bits), 2) % len(rainbow_colors)
            return rainbow_colors[idx]
        else:
            raise ValueError("Color map and rainbow_colors missing for large bitstrings")
    elif rainbow_colors is not None:
        idx = int(''.join(str(b) for b in bits), 2) % len(rainbow_colors)
        return rainbow_colors[idx]
    else:
        raise ValueError("No color_map or rainbow_colors provided for large bitstrings")

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
            bits = [int(b) for b in cell]
            lattice_row.append(bits)
        lattice.append(lattice_row)
    lattice_array = np.array(lattice)
    return step, lattice_array

def get_last_line_from_file(filename):
    """Get the last non-empty line from a file."""
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return None
        f.seek(-1, 2)
        lines = []
        while f.tell() > 0:
            char = f.read(1)
            if char == b'\n':
                if lines:
                    break
            f.seek(-2, 1)
            lines.append(char)
        if f.tell() == 0:
            f.seek(0)
            remaining = f.read().decode('utf-8')
            return remaining.strip().split('\n')[-1]
        last_line = b''.join(reversed(lines)).decode('utf-8').strip()
        return last_line if last_line else None

def load_final_lattices_alpha_mu_raster(L, B, gamma):
    """Load the final lattice from each file matching the specified parameters from rasterscanMu folder."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscanMu/L_{L}_B_{B}/g_{gamma}_a_*_mu_*.tsv")
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    print(f"Found {len(files)} files in rasterscanMu/L_{L}_B_{B} folder matching gamma={gamma}")
    lattice_data = {}
    for filename in tqdm(files, desc="Loading final lattices"):
        gamma_file, alpha, mu = extract_params_from_filename(filename)
        if gamma_file is None or alpha is None or mu is None:
            continue
        if gamma_file != gamma:
            continue
        try:
            last_line = get_last_line_from_file(filename)
            if last_line is None:
                continue
            step, lattice = parse_lattice_line(last_line)
            if lattice is not None:
                lattice_data[(alpha, mu)] = {
                    'lattice': lattice,
                    'step': step,
                    'L': L,
                    'B': B,
                    'gamma': gamma_file,
                    'alpha': alpha,
                    'mu': mu
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    return lattice_data

def create_lattice_alpha_mu_raster_plot(lattice_data, L, B, gamma):
    """Create a grid plot of lattices organized by mu (x-axis) and alpha (y-axis)."""
    if not lattice_data:
        print("No lattice data to plot")
        return
    alphas = sorted(set(key[0] for key in lattice_data.keys()))
    mus = sorted(set(key[1] for key in lattice_data.keys()))
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    color_map = None
    rainbow_colors = None
    if B > 4 and B <= 16:
        color_map = create_bitstring_color_map(B)
    if B > 16:
        rainbow_colors = get_rainbow_color_list(n_colors=65536, seed=42)
    alphas_reversed = list(reversed(alphas))
    fig, axes = plt.subplots(len(alphas), len(mus), figsize=(3*len(mus), 3*len(alphas)))
    if len(alphas) == 1 and len(mus) == 1:
        axes = [[axes]]
    elif len(alphas) == 1:
        axes = [axes]
    elif len(mus) == 1:
        axes = [[ax] for ax in axes]
    for i, alpha in enumerate(alphas_reversed):
        for j, mu in enumerate(mus):
            ax = axes[i][j]
            if (alpha, mu) in lattice_data:
                data = lattice_data[(alpha, mu)]
                lattice = data['lattice']
                step = data['step']
                L_current = data['L']
                rgb_img = np.zeros((L_current, L_current, 3))
                for x in range(L_current):
                    for y in range(L_current):
                        rgb_img[x, y] = bitstring_to_color(lattice[x, y], color_map=color_map, rainbow_colors=rainbow_colors)
                ax.imshow(rgb_img, interpolation='nearest')
                ax.set_title(f'α={alpha}, μ={mu:.4f}\nStep {step}', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'α={alpha}, μ={mu:.4f}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'Final Lattice States: Alpha vs Mu Raster\n(L={L}, B={B}, γ={gamma})', fontsize=16)
    fig.text(0.5, 0.02, 'Mu (Mutation Rate)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    output_dir = "src/understandabilityVsHamming2D/stochasticCommutable/plots/latticeGridMu"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/alpha_mu_raster_lattices_L_{L}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fname}")

def main_alpha_mu_raster(L=256, B=16, gamma=1):
    print(f"Looking for files with parameters: L={L}, B={B}, gamma={gamma} in rasterscanMu folder")
    lattice_data = load_final_lattices_alpha_mu_raster(L, B, gamma)
    if not lattice_data:
        print("No valid lattice data found.")
        return
    print(f"Successfully loaded {len(lattice_data)} lattices")
    create_lattice_alpha_mu_raster_plot(lattice_data, L, B, gamma)

if __name__ == "__main__":
    main_alpha_mu_raster(L=256, B=32, gamma=1)