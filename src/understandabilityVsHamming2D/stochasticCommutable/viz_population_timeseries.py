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

def int_to_bitstring(value, B):
    """Convert an integer back to bitstring for display."""
    return format(value, f'0{B}b')

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

def analyze_population_timeseries(filename):
    """
    Analyze population counts for each language over time.
    Returns timesteps, populations (dict of lang_int -> list), and B.
    """
    timesteps = []
    populations = {}
    B = None

    with open(filename, "r") as f:
        for line in tqdm(f, desc="Reading timeseries"):
            line = line.strip()
            if not line:
                continue
            step, lattice = parse_lattice_line(line)
            if lattice is None:
                continue
            timesteps.append(step)
            flat = [bitstring_to_int(bitstring) for row in lattice for bitstring in row]
            if B is None and flat:
                B = len(bin(max(flat))) - 2
            counts = Counter(flat)
            populations[step] = counts

    return timesteps, populations, B

def plot_population_timeseries(L, B, gamma, alpha, mu, max_langs=20, subdir="rasterscanMu"):
    """Plot population timeseries for all languages in a specific parameter file."""
    filename = find_file_for_params(L, B, gamma, alpha, mu, subdir=subdir)
    if filename is None:
        print(f"No file found for L={L}, B={B}, gamma={gamma}, alpha={alpha}, mu={mu} in {subdir}")
        return

    print(f"Found file: {filename}")

    timesteps, populations, B_detected = analyze_population_timeseries(filename)
    if not timesteps:
        print("No valid data found in file")
        return

    # Find all languages that ever appear
    all_langs = set()
    for counts in populations.values():
        all_langs.update(counts.keys())
    # Optionally, limit to most common
    lang_totals = Counter()
    for counts in populations.values():
        lang_totals.update(counts)
    top_langs = [lang for lang, _ in lang_totals.most_common(max_langs)]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_langs)))
    for i, lang_int in enumerate(top_langs):
        pop = [populations[step].get(lang_int, 0) for step in timesteps]
        bitstring_label = int_to_bitstring(lang_int, B)
        ax.plot(timesteps, pop, label=bitstring_label, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Population Count')
    ax.set_title(f'Language Population Timeseries\nL={L}, B={B}, γ={gamma}, α={alpha}, μ={mu}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

    output_dir = f"src/understandabilityVsHamming2D/stochasticCommutable/plots/populationTimeseries/{subdir}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/population_timeseries_L_{L}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    print(f"Plot saved to: {fname}")

def main():
    L = 256
    B = 16
    gamma = 1
    alpha = 1.2
    mu = 0.00001
    subdir = "long"
    plot_population_timeseries(L, B, gamma, alpha, mu, subdir=subdir)

if __name__ == "__main__":
    main()