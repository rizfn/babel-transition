import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances

def plot_multiple_files(files_to_plot, hamming_threshold=1):
    """Plot histograms for the specified files."""
    # Create subplots
    fig, axes = plt.subplots(1, len(files_to_plot), figsize=(6*len(files_to_plot), 6))
    if len(files_to_plot) == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    for idx, (gamma, alpha, N, B, mu) in enumerate(files_to_plot):
        # Load language data
        base_dir = f"src/understandabilityVsHammingSmall/outputs/top50/B_{B}/languages"
        languages_file = f"{base_dir}/g_{gamma}_a_{alpha}_N_{N}_B_{B}_mu_{mu}.tsv"

        if not os.path.exists(languages_file):
            print(f"Error: File {languages_file} not found.")
            continue

        df = pd.read_csv(languages_file, sep='\t', dtype={'language': str, 'generation': int})
        
        all_communicable_fractions = []
        all_hamming_distances = []
        generations = sorted(df['generation'].unique())
        
        print(f"Processing {len(generations)} generations for γ={gamma}, α={alpha}")
        
        # Process each generation
        for generation in generations:
            df_gen = df[df['generation'] == generation]
            
            # Convert language strings to binary arrays
            languages = np.array([[int(bit) for bit in str(lang_str)] for lang_str in df_gen['language']])
            
            # Compute Hamming distance matrix
            hamming_matrix = pairwise_distances(languages, metric='hamming') * B
            
            # Collect all pairwise distances (excluding diagonal)
            upper_tri_indices = np.triu_indices_from(hamming_matrix, k=1)
            all_hamming_distances.extend(hamming_matrix[upper_tri_indices])
            
            # For each agent, count how many agents are within threshold (including self)
            communicable_counts = np.sum(hamming_matrix <= hamming_threshold, axis=1)
            communicable_fractions = communicable_counts / len(languages)
            
            all_communicable_fractions.extend(communicable_fractions)
        
        # Calculate average hamming distance
        avg_hamming = np.mean(all_hamming_distances)
        
        # Plot histogram on the appropriate subplot
        bins = np.linspace(0, 1, 41)
        axes[idx].hist(all_communicable_fractions, bins=bins, edgecolor='black', alpha=0.6)
        axes[idx].set_xlabel('Fraction of population communicable (Hamming ≤ %d)' % hamming_threshold)
        axes[idx].set_ylabel('Number of agents')
        axes[idx].set_title(f'γ={gamma}, α={alpha}\n(Avg Hamming: {avg_hamming:.2f})')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribution of Communicable Fraction\n(N={N}, B={B}, μ={mu}, threshold={hamming_threshold})')
    plt.tight_layout()

    output_dir = "src/understandabilityVsHammingSmall/plots/numberCommunicable"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/communicable_hist_comparison_all_gens_B_{B}_thresh_{hamming_threshold}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

if __name__ == "__main__":
    # Define files to plot in one place
    files_to_plot = [
        (3, 1, 1000, 4, 0.01),
        (3, 2, 1000, 4, 0.01),
        (3, 3, 1000, 4, 0.01)
    ]
    
    # Plot comparison
    plot_multiple_files(files_to_plot, hamming_threshold=1)