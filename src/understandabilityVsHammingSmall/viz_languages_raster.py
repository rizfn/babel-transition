import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import re
from sklearn.metrics import pairwise_distances

def count_surviving_languages(languages, L, N, threshold_scale=0.5):
    """Count number of surviving languages based on abundance threshold."""
    # Count occurrences of each unique language
    unique_langs, counts = np.unique(languages, return_counts=True)

    lang_fractions = counts / N
    
    # Calculate threshold
    max_possible_languages = 2**L
    threshold = threshold_scale / (max_possible_languages)
    
    # Count languages above threshold
    surviving_count = np.sum(lang_fractions >= threshold)
    
    return surviving_count

def compute_avg_hamming_distance(languages, L):
    """Compute average pairwise Hamming distance between languages."""
    if len(languages) < 2:
        return 0.0
    
    # Convert language strings to binary arrays
    lang_arrays = []
    for lang_str in languages:
        lang_array = np.array([int(bit) for bit in str(lang_str)])
        lang_arrays.append(lang_array)
    
    lang_arrays = np.array(lang_arrays)
    
    # Compute pairwise Hamming distances
    hamming_matrix = pairwise_distances(lang_arrays, metric='hamming') * L
    
    # Get average (excluding diagonal)
    n = len(languages)
    avg_hamming = np.sum(hamming_matrix) / (n * (n - 1))
    
    return avg_hamming

def extract_params_from_filename(filename):
    """Extract parameters from filename like g_-1.0_a_1.0_N_1000_L_4_mu_0.01.tsv"""
    pattern = r'g_([^_]+)_a_([^_]+)_N_([^_]+)_L_([^_]+)_mu_([^_]+)\.tsv'
    match = re.match(pattern, filename)
    
    if match:
        g = float(match.group(1))
        a = float(match.group(2))
        N = int(match.group(3))
        L = int(match.group(4))
        mu = float(match.group(5))
        return g, a, N, L, mu
    else:
        return None

def process_all_files_timesteps(input_dir, threshold_scale=0.5):
    """Process all .tsv files and extract data for all timesteps."""
    all_data = {}
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.tsv'):
            params = extract_params_from_filename(filename)
            if params is None:
                continue
                
            g, a, N, L, mu = params
            filepath = os.path.join(input_dir, filename)
            
            try:
                df = pd.read_csv(filepath, sep="\t", dtype={'language': str, 'generation': int})
                
                # Process each generation
                for generation in df['generation'].unique():
                    df_step = df[df['generation'] == generation]
                    
                    if len(df_step) == 0:
                        continue
                    
                    languages = df_step['language'].tolist()
                    surviving_count = count_surviving_languages(languages, L, N, threshold_scale)
                    avg_hamming = compute_avg_hamming_distance(languages, L)
                    
                    if generation not in all_data:
                        all_data[generation] = []
                    
                    all_data[generation].append({
                        'g': g,
                        'a': a,
                        'N': N,
                        'L': L,
                        'mu': mu,
                        'surviving_languages': surviving_count,
                        'avg_hamming': avg_hamming,
                        'step': generation
                    })
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return all_data

def plot_frame(results, step, output_dir, threshold_scale):
    """Create dual heatmap for a single timestep."""
    if not results:
        print(f"No results to plot for step {step}")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results)
    
    # Get unique values for each parameter
    g_values = sorted(df['g'].unique())
    a_values = sorted(df['a'].unique(), reverse=True)  # Reverse alpha for proper ordering
    
    # Create pivot tables
    pivot_surviving = df.pivot_table(values='surviving_languages', 
                                   index='a',  # alpha on y-axis
                                   columns='g',  # gamma on x-axis
                                   aggfunc='mean')  # Use mean if multiple values
    
    pivot_hamming = df.pivot_table(values='avg_hamming', 
                                 index='a',  # alpha on y-axis
                                 columns='g',  # gamma on x-axis
                                 aggfunc='mean')  # Use mean if multiple values
    
    # Reindex to match the reversed alpha order
    pivot_surviving = pivot_surviving.reindex(a_values)
    pivot_hamming = pivot_hamming.reindex(a_values)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # First subplot: Number of surviving languages
    im1 = ax1.imshow(pivot_surviving.values, cmap='viridis', aspect='auto')
    
    # Set ticks and labels for first subplot
    ax1.set_xticks(range(len(g_values)))
    ax1.set_xticklabels([f'{g:.2f}' for g in g_values])
    ax1.set_yticks(range(len(a_values)))
    ax1.set_yticklabels([f'{a:.2f}' for a in a_values])
    
    # Add colorbar for first subplot
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Number of Surviving Languages')
    
    # Add value annotations for first subplot
    for i in range(len(a_values)):
        for j in range(len(g_values)):
            if not np.isnan(pivot_surviving.iloc[i, j]):
                text = ax1.text(j, i, f'{int(pivot_surviving.iloc[i, j])}',
                             ha="center", va="center", color="white", fontweight='bold')
    
    ax1.set_xlabel('Gamma (γ)')
    ax1.set_ylabel('Alpha (α)')
    ax1.set_title(f'Number of Surviving Languages\n(Threshold scale: {threshold_scale})')
    
    # Second subplot: Average Hamming distance
    im2 = ax2.imshow(pivot_hamming.values, cmap='plasma', aspect='auto')
    
    # Set ticks and labels for second subplot
    ax2.set_xticks(range(len(g_values)))
    ax2.set_xticklabels([f'{g:.2f}' for g in g_values])
    ax2.set_yticks(range(len(a_values)))
    ax2.set_yticklabels([f'{a:.2f}' for a in a_values])
    
    # Add colorbar for second subplot
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance')
    
    # Add value annotations for second subplot
    for i in range(len(a_values)):
        for j in range(len(g_values)):
            if not np.isnan(pivot_hamming.iloc[i, j]):
                text = ax2.text(j, i, f'{pivot_hamming.iloc[i, j]:.2f}',
                             ha="center", va="center", color="white", fontweight='bold')
    
    ax2.set_xlabel('Gamma (γ)')
    ax2.set_ylabel('Alpha (α)')
    ax2.set_title('Average Pairwise Hamming Distance')
    
    plt.suptitle(f"Step {step}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save frame
    os.makedirs(output_dir, exist_ok=True)
    outname = os.path.join(output_dir, f"frame_{step:04d}.png")
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Frame saved: {outname}")

def compute_averages_across_frames(all_data):
    """Compute averages across all frames for each parameter combination."""
    # Collect all data points across all timesteps
    all_results = []
    for step_data in all_data.values():
        all_results.extend(step_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Group by parameter combinations and compute averages
    grouped = df.groupby(['g', 'a', 'N', 'L', 'mu']).agg({
        'surviving_languages': 'mean',
        'avg_hamming': 'mean'
    }).reset_index()
    
    return grouped.to_dict('records')

def plot_averages(average_results, output_path, threshold_scale):
    """Create dual heatmap of averages across all frames."""
    if not average_results:
        print("No average results to plot")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(average_results)
    
    # Get unique values for each parameter
    g_values = sorted(df['g'].unique())
    a_values = sorted(df['a'].unique(), reverse=True)  # Reverse alpha for proper ordering
    
    # Create pivot tables
    pivot_surviving = df.pivot_table(values='surviving_languages', 
                                   index='a',  # alpha on y-axis
                                   columns='g',  # gamma on x-axis
                                   aggfunc='mean')  # Use mean if multiple values
    
    pivot_hamming = df.pivot_table(values='avg_hamming', 
                                 index='a',  # alpha on y-axis
                                 columns='g',  # gamma on x-axis
                                 aggfunc='mean')  # Use mean if multiple values
    
    # Reindex to match the reversed alpha order
    pivot_surviving = pivot_surviving.reindex(a_values)
    pivot_hamming = pivot_hamming.reindex(a_values)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # First subplot: Number of surviving languages
    im1 = ax1.imshow(pivot_surviving.values, cmap='viridis', aspect='auto')
    
    # Set ticks and labels for first subplot
    ax1.set_xticks(range(len(g_values)))
    ax1.set_xticklabels([f'{g:.2f}' for g in g_values])
    ax1.set_yticks(range(len(a_values)))
    ax1.set_yticklabels([f'{a:.2f}' for a in a_values])
    
    # Add colorbar for first subplot
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Average Number of Surviving Languages')
    
    # Add value annotations for first subplot
    for i in range(len(a_values)):
        for j in range(len(g_values)):
            if not np.isnan(pivot_surviving.iloc[i, j]):
                text = ax1.text(j, i, f'{pivot_surviving.iloc[i, j]:.1f}',
                             ha="center", va="center", color="white", fontweight='bold')
    
    ax1.set_xlabel('Gamma (γ)')
    ax1.set_ylabel('Alpha (α)')
    ax1.set_title(f'Average Number of Surviving Languages\n(Threshold scale: {threshold_scale})')
    
    # Second subplot: Average Hamming distance
    im2 = ax2.imshow(pivot_hamming.values, cmap='plasma', aspect='auto')
    
    # Set ticks and labels for second subplot
    ax2.set_xticks(range(len(g_values)))
    ax2.set_xticklabels([f'{g:.2f}' for g in g_values])
    ax2.set_yticks(range(len(a_values)))
    ax2.set_yticklabels([f'{a:.2f}' for a in a_values])
    
    # Add colorbar for second subplot
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Hamming Distance')
    
    # Add value annotations for second subplot
    for i in range(len(a_values)):
        for j in range(len(g_values)):
            if not np.isnan(pivot_hamming.iloc[i, j]):
                text = ax2.text(j, i, f'{pivot_hamming.iloc[i, j]:.2f}',
                             ha="center", va="center", color="white", fontweight='bold')
    
    ax2.set_xlabel('Gamma (γ)')
    ax2.set_ylabel('Alpha (α)')
    ax2.set_title('Average Pairwise Hamming Distance')
    
    plt.suptitle("Averages Across All Frames")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Average plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create video frames and average plot of surviving languages heatmaps')
    parser.add_argument("--input_dir", type=str, 
                       default="src/understandabilityVsHammingSmall/outputs/top50/languages/",
                       help="Directory containing language .tsv files")
    parser.add_argument("--frames_output_dir", type=str, 
                       default="src/understandabilityVsHammingSmall/plots/languages/frames/",
                       help="Output directory for frames")
    parser.add_argument("--average_output", type=str, 
                       default="src/understandabilityVsHammingSmall/plots/languages/average_heatmap.png",
                       help="Output path for average heatmap")
    parser.add_argument("--threshold_scale", type=float, default=0.5,
                       help="Threshold scaling factor (default: 0.5)")
    
    args = parser.parse_args()
    
    # Process all files and get data for all timesteps
    print(f"Processing files in {args.input_dir}")
    all_data = process_all_files_timesteps(args.input_dir, args.threshold_scale)
    
    print(f"Found data for {len(all_data)} timesteps")
    
    # Create frames for each timestep
    for step in sorted(all_data.keys()):
        print(f"Creating frame for step {step}")
        plot_frame(all_data[step], step, args.frames_output_dir, args.threshold_scale)
    
    # Compute averages across all frames
    print("Computing averages across all frames...")
    average_results = compute_averages_across_frames(all_data)
    
    # Create average plot
    print("Creating average plot...")
    plot_averages(average_results, args.average_output, args.threshold_scale)

if __name__ == "__main__":
    main()