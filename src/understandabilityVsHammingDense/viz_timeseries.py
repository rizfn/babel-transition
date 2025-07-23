import os
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from tqdm import tqdm
import pandas as pd

def extract_params_from_filename(filename):
    """Extract gamma, alpha, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    if gamma and alpha and mu:
        return (float(gamma.group(1)), float(alpha.group(1)), float(mu.group(1)))
    return (None,)*3

def bitstring_to_int(bitstring):
    """Convert a bitstring to an integer for easier processing."""
    return int(bitstring, 2)

def int_to_bitstring(value, B):
    """Convert an integer back to bitstring for display."""
    return format(value, f'0{B}b')

def find_file_for_params(gamma, alpha, N, B, mu, subdir="top50"):
    """Find the file corresponding to given parameters in the specified subdir."""
    pattern = os.path.join(
        os.path.dirname(__file__),
        f"outputs/{subdir}/languages/g_{gamma}_a_{alpha}_N_{N}_B_{B}_mu_{mu}.tsv"
    )
    if os.path.exists(pattern):
        return pattern
    
    # Try to find file with scientific notation for mu
    folder = os.path.join(
        os.path.dirname(__file__),
        f"outputs/{subdir}/languages"
    )
    if not os.path.exists(folder):
        return None
    
    for fname in os.listdir(folder):
        if re.fullmatch(rf"g_{gamma}_a_{alpha}_N_{N}_B_{B}_mu_.*\.tsv", fname):
            mu_match = re.search(r'mu_([0-9eE\.\+-]+)', fname)
            if mu_match:
                mu_in_file = float(mu_match.group(1))
                if f"mu_{mu}" in fname or np.isclose(float(mu), mu_in_file, rtol=1e-8):
                    return os.path.join(folder, fname)
    return None

def analyze_population_timeseries(filename):
    """
    Analyze population counts for each language over time from mean-field model output.
    Returns timesteps, populations (dict of lang_int -> list), and B.
    """
    try:
        df = pd.read_csv(filename, sep='\t', dtype={'language': str})
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], {}, None
    
    if 'generation' not in df.columns or 'language' not in df.columns:
        print(f"Required columns not found in {filename}")
        return [], {}, None
    
    # Get unique generations and sort them
    generations = sorted(df['generation'].unique())
    
    # Initialize populations dictionary
    populations = {}
    B = None
    
    print(f"Processing {len(generations)} generations...")
    
    for gen in tqdm(generations, desc="Processing generations"):
        gen_data = df[df['generation'] == gen]
        
        # Convert language strings to integers
        lang_ints = []
        for lang_str in gen_data['language']:
            lang_str = str(lang_str).strip()
            if B is None:
                B = len(lang_str)
            try:
                lang_int = bitstring_to_int(lang_str)
                lang_ints.append(lang_int)
            except ValueError:
                print(f"Invalid language string: {lang_str}")
                continue
        
        # Count occurrences
        counts = Counter(lang_ints)
        populations[gen] = counts
    
    return generations, populations, B

def plot_population_timeseries(gamma, alpha, N, B, mu, max_langs=60, subdir="top50"):
    """Plot population timeseries for all languages in a specific parameter file."""
    filename = find_file_for_params(gamma, alpha, N, B, mu, subdir=subdir)
    if filename is None:
        print(f"No file found for gamma={gamma}, alpha={alpha}, N={N}, B={B}, mu={mu} in {subdir}")
        return

    print(f"Found file: {filename}")

    timesteps, populations, B_detected = analyze_population_timeseries(filename)
    if not timesteps:
        print("No valid data found in file")
        return

    # Use detected B if available
    if B_detected is not None:
        B = B_detected

    # Find all languages that ever appear
    all_langs = set()
    for counts in populations.values():
        all_langs.update(counts.keys())
    
    # Optionally, limit to most common languages
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

    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Count')
    ax.set_title(f'Language Population Timeseries\nN={N}, B={B}, γ={gamma}, α={alpha}, μ={mu}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)

    output_dir = f"src/understandabilityVsHammingDense/plots/populationTimeseries/{subdir}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/population_timeseries_N_{N}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()
    print(f"Plot saved to: {fname}")

def main():
    gamma = 1
    alpha = 0.4
    N = 1000
    B = 16
    mu = 0.000001
    subdir = "timeseries"
    plot_population_timeseries(gamma, alpha, N, B, mu, subdir=subdir)

if __name__ == "__main__":
    main()