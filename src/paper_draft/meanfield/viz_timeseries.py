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

def find_file_for_params(gamma, alpha, N, B, mu, sim=0):
    """Find the file corresponding to given parameters."""
    pattern = os.path.join(
        os.path.dirname(__file__),
        f"outputs/timeseries/N_{N}_B_{B}/g_{gamma}_a_{alpha}_mu_{mu}_sim_{sim}.tsv"
    )
    if os.path.exists(pattern):
        return pattern
    
    # Try to find file with scientific notation for mu
    folder = os.path.join(
        os.path.dirname(__file__),
        f"outputs/timeseries/N_{N}_B_{B}"
    )
    if not os.path.exists(folder):
        return None
    
    for fname in os.listdir(folder):
        if re.fullmatch(rf"g_{gamma}_a_{alpha}_mu_.*_sim_{sim}\.tsv", fname):
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
        df = pd.read_csv(filename, sep='\t', dtype={'language': str, 'population': int})
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], {}, None
    
    if 'generation' not in df.columns or 'language' not in df.columns or 'population' not in df.columns:
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
        
        # Build dictionary from language string to population count
        counts = {}
        for _, row in gen_data.iterrows():
            lang_str = str(row['language']).strip()
            if B is None:
                B = len(lang_str)
            try:
                lang_int = bitstring_to_int(lang_str)
                counts[lang_int] = int(row['population'])
            except ValueError:
                print(f"Invalid language string: {lang_str}")
                continue
        
        populations[gen] = counts
        
        # Debug: print total population for first few generations
        if gen <= 953:
            total_pop = sum(counts.values())
            print(f"Generation {gen}: Total population = {total_pop}")
    
    return generations, populations, B

def plot_population_timeseries(gamma, alpha, N, B, mu, max_langs=200, sim=0):
    """Plot population timeseries for all languages in a specific parameter file."""
    filename = find_file_for_params(gamma, alpha, N, B, mu, sim=sim)
    if filename is None:
        print(f"No file found for gamma={gamma}, alpha={alpha}, N={N}, B={B}, mu={mu}, sim={sim}")
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

    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='none')
    ax.patch.set_facecolor('none')
    
    # Generate rainbow colors uniformly distributed and shuffled
    colors = plt.cm.rainbow(np.linspace(0, 1, len(top_langs)))
    np.random.shuffle(colors)
    
    for i, lang_int in enumerate(top_langs):
        pop = [populations[step].get(lang_int, 0) / N for step in timesteps]  # Normalize by N
        bitstring_label = int_to_bitstring(lang_int, B)
        ax.plot(timesteps, pop, label=bitstring_label, color=colors[i], linewidth=2)

    # Set ticks - only show first and last
    ax.set_xticks([timesteps[0], timesteps[-1]])
    ax.set_xticklabels([f'{timesteps[0]}', f'{timesteps[-1]}'], fontsize=30)
    
    # Set y-axis ticks with 0 as minimum
    y_min, y_max = ax.get_ylim()
    y_min = max(0, y_min)  # Ensure minimum is 0
    ax.set_yticks([y_min, y_max])
    ax.set_yticklabels([f'{y_min:.2f}', f'{y_max:.2f}'], fontsize=30)

    # Add minor grid lines
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.1, which='minor')
    ax.minorticks_on()

    ax.set_xlabel('Generation', fontsize=40, labelpad=-25)
    ax.set_ylabel('Population Fraction', fontsize=40, labelpad=-55)

    output_dir = f"src/paper_draft/meanfield/plots/populationTimeseries"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/popTimeseries_N_{N}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}_sim_{sim}.pdf"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    plt.close()
    print(f"Plot saved to: {fname}")

def main():
    gamma = 1
    alpha = 0.4
    N = 65536
    B = 16
    mu = 0.00001
    sim = 0
    plot_population_timeseries(gamma, alpha, N, B, mu, sim=sim)

if __name__ == "__main__":
    main()