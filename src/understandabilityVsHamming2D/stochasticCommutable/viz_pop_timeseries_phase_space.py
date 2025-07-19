import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from collections import Counter
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def analyze_population_matrix(filename):
    """
    Analyze population counts for each language over time.
    Returns timesteps, population_matrix (timesteps x languages), and language_mapping.
    """
    timesteps = []
    populations = {}
    all_languages = set()

    # First pass: collect all data and find all languages
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
            counts = Counter(flat)
            populations[step] = counts
            all_languages.update(counts.keys())

    # Create language mapping and population matrix
    sorted_languages = sorted(all_languages)
    lang_to_idx = {lang: i for i, lang in enumerate(sorted_languages)}
    
    population_matrix = np.zeros((len(timesteps), len(sorted_languages)))
    
    for t_idx, step in enumerate(timesteps):
        for lang, count in populations[step].items():
            lang_idx = lang_to_idx[lang]
            population_matrix[t_idx, lang_idx] = count

    return timesteps, population_matrix, sorted_languages

def plot_phase_space_trajectory(L, B, gamma, alpha, mu, subdir="rasterscanMu"):
    """Plot 3D phase space trajectory of population dynamics."""
    filename = find_file_for_params(L, B, gamma, alpha, mu, subdir=subdir)
    if filename is None:
        print(f"No file found for L={L}, B={B}, gamma={gamma}, alpha={alpha}, mu={mu} in {subdir}")
        return

    print(f"Found file: {filename}")

    timesteps, population_matrix, languages = analyze_population_matrix(filename)
    if len(timesteps) == 0:
        print("No valid data found in file")
        return

    print(f"Population matrix shape: {population_matrix.shape}")
    print(f"Number of unique languages: {len(languages)}")

    # Standardize the data
    scaler = StandardScaler()
    population_scaled = scaler.fit_transform(population_matrix)

    # Perform PCA to reduce to 3 components
    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(population_scaled)

    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create color map based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))

    # Plot trajectory as connected line segments with color coding
    for i in range(len(timesteps) - 1):
        ax.plot3D([trajectory_3d[i, 0], trajectory_3d[i+1, 0]], 
                  [trajectory_3d[i, 1], trajectory_3d[i+1, 1]], 
                  [trajectory_3d[i, 2], trajectory_3d[i+1, 2]], 
                  color=colors[i], linewidth=2, alpha=0.0)

    # Add scatter points to show individual timesteps
    scatter = ax.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 
                        c=timesteps, cmap='viridis', s=20, alpha=0.6)

    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
    ax.set_title(f'Population Dynamics Phase Space Trajectory\nL={L}, B={B}, γ={gamma}, α={alpha}, μ={mu}')

    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time Step')

    # Add legend
    ax.legend()

    # Save plot
    output_dir = f"src/understandabilityVsHamming2D/stochasticCommutable/plots/phaseSpace/{subdir}"
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/phase_space_trajectory_L_{L}_B_{B}_g_{gamma}_a_{alpha}_mu_{mu}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    print(f"Plot saved to: {fname}")

def main():
    L = 256
    B = 16
    gamma = 1
    alpha = 0.8
    mu = 0.005
    subdir = "long"
    plot_phase_space_trajectory(L, B, gamma, alpha, mu, subdir=subdir)

if __name__ == "__main__":
    main()