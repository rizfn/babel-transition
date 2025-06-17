import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances

def compute_avg_hamming_per_generation(gamma, alpha, N, L, mu):
    """Compute average hamming distance for each generation."""
    # Load language data
    base_dir = "src/understandabilityVsHammingSmall/outputs/top50/languages"
    languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_L_{L}_mu_{mu}.tsv"

    if not os.path.exists(languages_file):
        print(f"Error: File {languages_file} not found.")
        return None, None

    df = pd.read_csv(languages_file, sep='\t', dtype={'language': str, 'generation': int})
    
    generations = sorted(df['generation'].unique())
    avg_hamming_per_gen = []
    
    print(f"Processing {len(generations)} generations for γ={gamma}, α={alpha}")
    
    # Process each generation
    for generation in generations:
        df_gen = df[df['generation'] == generation]
        
        # Convert language strings to binary arrays
        languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df_gen['language']])
        
        # Compute Hamming distance matrix
        hamming_matrix = pairwise_distances(languages, metric='hamming') * L
        
        # Get average pairwise distance (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(hamming_matrix, k=1)
        avg_hamming = np.mean(hamming_matrix[upper_tri_indices])
        
        avg_hamming_per_gen.append(avg_hamming)
    
    return generations, avg_hamming_per_gen

def plot_hamming_timeseries(files_to_plot):
    """Plot average hamming distance timeseries for multiple files."""    
    
    for idx, (gamma, alpha, N, L, mu) in enumerate(files_to_plot):
        generations, avg_hamming_per_gen = compute_avg_hamming_per_generation(gamma, alpha, N, L, mu)
        
        if generations is None or avg_hamming_per_gen is None:
            continue
        
        plt.plot(generations, avg_hamming_per_gen, label=f'γ={gamma}, α={alpha}', alpha=0.8)
    
    plt.xlabel('Generation')
    plt.ylabel('Average Hamming Distance')
    plt.title(f'Average Hamming Distance Over Time\n(N={N}, L={L}, μ={mu})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = "src/understandabilityVsHammingSmall/plots/avgHammingTimeseries"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/avg_hamming_timeseries_N_{N}_L_{L}_mu_{mu}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timeseries plot saved to: {output_file}")

if __name__ == "__main__":
    # Define files to plot in one place
    files_to_plot = [
        (3, 0, 1000, 4, 0.01),
        (3, 1, 1000, 4, 0.01),
        (3, 2, 1000, 4, 0.01),
        (3, 3, 1000, 4, 0.01)
    ]
    
    # Plot timeseries
    plot_hamming_timeseries(files_to_plot)