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

def parse_population_line(line):
    """Parse a single line from population timeseries format."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    
    step = int(parts[0])
    population_counts = {}
    
    # Parse each bitstring,count pair
    for i in range(1, len(parts)):
        if ',' in parts[i]:
            bitstring, count_str = parts[i].split(',')
            population_counts[bitstring] = int(count_str)
    
    return step, population_counts

def bitstring_to_int(bitstring):
    """Convert a bitstring to an integer for easier processing."""
    return int(bitstring, 2)

def int_to_bitstring(value, B):
    """Convert an integer back to bitstring for display."""
    return format(value, f'0{B}b')

def find_file_for_params(L, B, gamma, alpha, mu):
    """Find the file corresponding to given parameters."""
    pattern = os.path.join(
        os.path.dirname(__file__),
        f"outputs/populationTimeseries/L_{L}_B_{B}/g_{gamma}_a_{alpha}_mu_{mu}.tsv"
    )
    if os.path.exists(pattern):
        return pattern
    
    # Try to find file with scientific notation for mu
    folder = os.path.join(
        os.path.dirname(__file__),
        f"outputs/populationTimeseries/L_{L}_B_{B}"
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
    Returns timesteps, populations (dict of step -> dict of bitstring -> count), and B.
    """
    timesteps = []
    populations = {}
    B = None

    with open(filename, "r") as f:
        for line in tqdm(f, desc="Reading population timeseries"):
            line = line.strip()
            if not line:
                continue
            
            step, pop_counts = parse_population_line(line)
            if pop_counts is None:
                continue
                
            timesteps.append(step)
            populations[step] = pop_counts
            
            # Determine B from the first bitstring we see
            if B is None and pop_counts:
                first_bitstring = next(iter(pop_counts.keys()))
                B = len(first_bitstring)

    return timesteps, populations, B

def plot_population_timeseries(L, B, gamma, alpha, mu, max_langs=20):
    """Plot population timeseries for all languages in a specific parameter file."""
    filename = find_file_for_params(L, B, gamma, alpha, mu)
    if filename is None:
        print(f"No file found for L={L}, B={B}, gamma={gamma}, alpha={alpha}, mu={mu}")
        return

    print(f"Found file: {filename}")

    timesteps, populations, B_detected = analyze_population_timeseries(filename)
    if not timesteps:
        print("No valid data found in file")
        return

    # Find all languages that ever appear and their total counts
    lang_totals = Counter()
    for pop_counts in populations.values():
        lang_totals.update(pop_counts)
    
    # Get top languages by total count
    top_langs = [bitstring for bitstring, _ in lang_totals.most_common(max_langs)]

    # Create shuffled rainbow colors
    np.random.seed(53)  # For reproducible colors
    cmap = plt.cm.rainbow
    colors = [cmap(i / (len(top_langs) - 1))[:3] for i in range(len(top_langs))]
    np.random.shuffle(colors)
    np.random.seed(None)  # Reset seed

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='none')
    ax.patch.set_facecolor('none')
    
    lattice_size = L * L
    
    for i, bitstring in enumerate(top_langs):
        pop = [populations[step].get(bitstring, 0) / lattice_size for step in timesteps]
        ax.plot(timesteps, pop, color=colors[i], linewidth=2)

    ax.set_xlabel('Time', fontsize=32, labelpad=-20)
    ax.set_ylabel('Population', fontsize=32, labelpad=-30)
    
    # Get max normalized population value for y-axis and round to 2 significant figures
    max_pop_raw = max(max(populations[step].values()) for step in timesteps if populations[step]) / lattice_size
    # Round to 2 significant figures
    max_pop = round(max_pop_raw, -int(np.floor(np.log10(abs(max_pop_raw)))) + 1)
    
    # Enable grid with both major and minor ticks
    ax.grid(True, which='both', alpha=0.3)
    ax.minorticks_on()
    
    # Set displayed ticks to only min and max values (this will override the minor ticks)
    ax.set_xticks([min(timesteps), max(timesteps)])
    ax.set_yticks([0, max_pop])
    
    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)

    output_dir = f"src/paper_draft/2D/plots/populationTimeseries"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/population_timeseries_L_{L}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}.svg"
    plt.savefig(fname, dpi=300, format='svg', bbox_inches='tight', facecolor='none', edgecolor='none', transparent=True)
    plt.tight_layout()
    print(f"Plot saved to: {fname}")

def main():
    L = 256
    B = 16
    gamma = 1
    alpha = 0.4
    mu = 1e-05
    max_langs = 40
    
    plot_population_timeseries(L, B, gamma, alpha, mu, max_langs)

if __name__ == "__main__":
    main()