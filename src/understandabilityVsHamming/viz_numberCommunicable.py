import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances

def number_communicable_histogram(gamma, alpha, N, L, mu, hamming_threshold=1):
    """
    For each agent, compute the fraction of the population it can communicate with
    (i.e., hamming distance <= hamming_threshold), and plot the histogram.
    """
    # Load language data
    base_dir = "src/understandabilityVsHamming/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str})

    # Use last generation if present
    if 'generation' in df.columns:
        last_gen = df['generation'].max()
        df = df[df['generation'] == last_gen]
    else:
        last_gen = None

    # Convert language strings to binary arrays
    languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df['language']])

    # Compute Hamming distance matrix
    hamming_matrix = pairwise_distances(languages, metric='hamming') * L

    # For each agent, count how many agents are within threshold (including self)
    communicable_counts = np.sum(hamming_matrix <= hamming_threshold, axis=1)
    communicable_fractions = communicable_counts / len(languages)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, 1, 41)
    plt.hist(communicable_fractions, bins=bins, edgecolor='black', alpha=0.6)
    plt.xlabel('Fraction of population communicable (Hamming ≤ %d)' % hamming_threshold)
    plt.ylabel('Number of agents')
    plt.title(f'Distribution of Communicable Fraction\n'
              f'(γ={gamma}, α={alpha}, N={N}, L={L}, μ={mu}, threshold={hamming_threshold})')
    plt.tight_layout()

    output_dir = "src/understandabilityVsHamming/plots/numberCommunicable"
    os.makedirs(output_dir, exist_ok=True)
    gen_suffix = f"_gen_{last_gen}" if last_gen is not None else ""
    output_file = f"{output_dir}/communicable_hist_g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}_thresh_{hamming_threshold}{gen_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    number_communicable_histogram(1, 0, 1000, 16, 0.01, hamming_threshold=1)