import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu, K from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9.]+)', filename)
    K = re.search(r'K_([0-9]+)', filename)
    if gamma and alpha and L and B and mu and K:
        return (float(gamma.group(1)), float(alpha.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)), int(K.group(1)))
    return (None,)*6

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
            # Each cell is a bitstring, convert to tuple for uniqueness
            bits = tuple(int(b) for b in cell)
            lattice_row.append(bits)
        lattice.append(lattice_row)
    
    return step, lattice

def bitstring_to_int(bitstring):
    """Convert a bitstring tuple to an integer for easier processing."""
    return sum(bit * (2 ** i) for i, bit in enumerate(reversed(bitstring)))

def int_to_bitstring(value, B):
    """Convert an integer back to bitstring for display."""
    return format(value, f'0{B}b')

def find_file_for_params(gamma, alpha):
    """Find the file corresponding to given gamma and alpha parameters."""
    pattern = os.path.join(os.path.dirname(__file__), "outputs/latticeTimeseries/rasterscan/L_*_g_*_a_*_B_*_mu_*_K_*.tsv")
    files = glob.glob(pattern)
    
    for filename in files:
        file_gamma, file_alpha, L, B, mu, K = extract_params_from_filename(filename)
        if file_gamma == gamma and file_alpha == alpha:
            return filename, L, B, mu, K
    
    return None, None, None, None, None

def analyze_population_timeseries(filename, top_n=10):
    """
    Analyze population counts for each language over time.
    Returns timesteps, population data for top N languages, and their bitstring representations.
    """
    # First pass: count total occurrences of each language across all timesteps
    total_counts = Counter()
    
    print("First pass: counting total language occurrences...")
    with open(filename, "r") as f:
        for line in tqdm(f, desc="Counting languages"):
            line = line.strip()
            if not line:
                continue
            
            step, lattice = parse_lattice_line(line)
            if lattice is None:
                continue
            
            # Count occurrences in this timestep
            for row in lattice:
                for bitstring in row:
                    lang_int = bitstring_to_int(bitstring)
                    total_counts[lang_int] += 1
    
    # Get the top N most common languages
    top_languages = [lang for lang, count in total_counts.most_common(top_n)]
    print(f"Top {top_n} languages by total occurrence: {top_languages}")
    
    # Second pass: track populations over time for top languages
    timesteps = []
    populations = defaultdict(list)  # populations[lang_int] = [count_at_t0, count_at_t1, ...]
    
    print("Second pass: tracking populations over time...")
    with open(filename, "r") as f:
        for line in tqdm(f, desc="Tracking populations"):
            line = line.strip()
            if not line:
                continue
            
            step, lattice = parse_lattice_line(line)
            if lattice is None:
                continue
            
            timesteps.append(step)
            
            # Count occurrences of each top language in this timestep
            step_counts = Counter()
            for row in lattice:
                for bitstring in row:
                    lang_int = bitstring_to_int(bitstring)
                    if lang_int in top_languages:
                        step_counts[lang_int] += 1
            
            # Record counts for all top languages (0 if not present)
            for lang in top_languages:
                populations[lang].append(step_counts.get(lang, 0))
    
    return timesteps, populations, top_languages

def plot_population_timeseries(gamma, alpha, top_n=10):
    """Plot population timeseries for the top N languages in a specific parameter file."""
    
    # Find the file for these parameters
    filename, L, B, mu, K = find_file_for_params(gamma, alpha)
    
    if filename is None:
        print(f"No file found for gamma={gamma}, alpha={alpha}")
        return
    
    print(f"Found file: {filename}")
    print(f"Parameters: L={L}, B={B}, mu={mu}, K={K}")
    
    # Analyze the timeseries
    timesteps, populations, top_languages = analyze_population_timeseries(filename, top_n)
    
    if not timesteps:
        print("No valid data found in file")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each language
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_languages)))
    
    for i, lang_int in enumerate(top_languages):
        bitstring_label = int_to_bitstring(lang_int, B)
        ax.plot(timesteps, populations[lang_int], 
                label=bitstring_label, color=colors[i], linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Population Count')
    ax.set_title(f'Language Population Timeseries\nγ={gamma}, α={alpha} (L={L}, B={B})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    output_dir = "src/understandabilityVsHamming2D/commutableHamming/plots/populationTimeseries"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/population_timeseries_g_{gamma}_a_{alpha}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    
    print(f"Plot saved to: {fname}")

def main():
    # Example usage - modify these parameters as needed
    gamma = 3.0  # Change this to your desired gamma value
    alpha = 0.0  # Change this to your desired alpha value
    top_n = 20   # Number of top languages to plot
    
    plot_population_timeseries(gamma, alpha, top_n)

if __name__ == "__main__":
    main()