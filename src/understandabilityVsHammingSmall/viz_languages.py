import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def bitstring_to_color(bits):    
    r = 0.3 + 0.7 * int(bits[0])
    g = 0.3 + 0.7 * int(bits[1]) 
    b = 0.3 + 0.7 * int(bits[2])
    return (r, g, b)

    
def plot_languages(languages, step, outname, L):
    """Plot languages as a heatmap with unique language clusters."""
    # Convert language strings to binary arrays
    langs_array = np.array([[int(bit) for bit in lang] for lang in languages])
    
    # Get unique languages and create labels
    unique_languages, inverse_indices = np.unique(langs_array, axis=0, return_inverse=True)
    labels = inverse_indices  # Each agent gets the label of their unique language
    
    # Create color mapping using your existing color function
    color_mapping = {}
    for i, unique_lang in enumerate(unique_languages):
        color_mapping[i] = bitstring_to_color(unique_lang)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Heatmap colored by unique language clusters
    sorted_indices = np.argsort(labels)
    sorted_languages = langs_array[sorted_indices]
    sorted_labels = labels[sorted_indices]
    language_colors = np.ones((len(sorted_languages), L, 3))
    
    for i, (language, label) in enumerate(zip(sorted_languages, sorted_labels)):
        color = color_mapping[label]
        for j, bit in enumerate(language):
            if bit == 1:
                language_colors[i, j] = color
    
    ax.imshow(language_colors, aspect='auto', interpolation='none',
              extent=[-0.5, L-0.5, len(sorted_languages)-0.5, -0.5])
    ax.set_title(f'Languages Heatmap (Step {step})')
    ax.set_xlabel('Bit Position')
    ax.set_ylabel('Language ID (sorted by language type)')
    
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

def main(g, a, N, L, mu, step):
    infile = f"src/understandabilityVsHammingSmall/outputs/top50/languages/g_{g}_a_{a}_N_{N}_L_{L}_mu_{mu}.tsv"
    outname = f"src/understandabilityVsHammingSmall/plots/languages/g_{g}_a_{a}_N_{N}_L_{L}_mu_{mu}.png"

    df = pd.read_csv(infile, sep="\t", dtype={'language': str, 'generation': int})
    if step is None:
        step = df['generation'].max()
    else:
        step = step
    df = df[df['generation'] == step]

    print(df)
    
    langs = df['language'].tolist()  # Convert to list of strings
    langs = np.array(langs)

    plot_languages(langs, step, outname, L)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=float, default=-1)
    parser.add_argument("--a", type=float, default=1)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--step", type=int, default=None, help="Which step to plot (default: last)")
    args = parser.parse_args()

    main(args.g, args.a, args.N, args.L, args.mu, args.step)
