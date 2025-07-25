import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
from itertools import product
from tqdm import tqdm

def extract_params_from_filename(filename):
    """Extract gamma, alpha, L, B, mu from filename."""
    gamma = re.search(r'g_([+-]?\d+\.?\d*)_', filename)
    alpha = re.search(r'a_([+-]?\d+\.?\d*)_', filename)
    L = re.search(r'L_([0-9]+)', filename)
    B = re.search(r'B_([0-9]+)', filename)
    mu = re.search(r'mu_([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)', filename)
    
    if gamma and alpha and L and B and mu:
        return (float(gamma.group(1)), float(alpha.group(1)),
                int(L.group(1)), int(B.group(1)), float(mu.group(1)))
    return (None,)*5

def create_bitstring_color_map(B, seed=42):
    """Create a fixed mapping from bitstrings to shuffled colors from rainbow colormap."""
    # Set random seed for reproducible color assignment
    np.random.seed(seed)
    
    # Get all possible bitstrings
    all_bitstrings = [tuple(bits) for bits in product([0, 1], repeat=B)]
    num_bitstrings = len(all_bitstrings)
    
    # Get colors from rainbow colormap
    cmap = plt.cm.rainbow
    colors = [cmap(i / (num_bitstrings - 1))[:3] for i in range(num_bitstrings)]  # Take only RGB, drop alpha
    
    # Shuffle colors to avoid similar bitstrings getting similar colors
    np.random.shuffle(colors)
    
    # Create mapping
    color_map = dict(zip(all_bitstrings, colors))
    
    # Reset random seed so it doesn't affect other random operations
    np.random.seed(None)
    
    return color_map

def get_rainbow_color_list(n_colors=65536, seed=42):
    """Get a shuffled list of colors from the rainbow colormap."""
    np.random.seed(seed)
    cmap = plt.cm.rainbow
    colors = [cmap(i / (n_colors - 1))[:3] for i in range(n_colors)]
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors

def bitstring_to_color(bits, color_map=None, rainbow_colors=None):
    """Map a bitstring to a unique RGB color."""
    if len(bits) == 4:
        r = 0.3 + 0.7 * bits[0]
        g = 0.3 + 0.7 * bits[1]
        b = 0.3 + 0.7 * bits[2]
        brightness = 0.3 + 0.5 * bits[3]
        color = np.array([r, g, b]) * brightness
        color = np.clip(color, 0, 1)
        return color
    elif len(bits) == 2:
        base = 0.2
        delta = 0.6
        r = base + delta * bits[0]
        g = base + delta * bits[1]
        b = base + delta * (bits[0] ^ bits[1])  # XOR for more separation
        return [r, g, b]
    elif color_map is not None:
        bitstring_tuple = tuple(bits)
        if bitstring_tuple in color_map:
            return color_map[bitstring_tuple]
        # fallback if not found
        if rainbow_colors is not None:
            idx = int(''.join(str(b) for b in bits), 2) % len(rainbow_colors)
            return rainbow_colors[idx]
        else:
            raise ValueError("Color map and rainbow_colors missing for large bitstrings")
    elif rainbow_colors is not None:
        idx = int(''.join(str(b) for b in bits), 2) % len(rainbow_colors)
        return rainbow_colors[idx]
    else:
        raise ValueError("No color_map or rainbow_colors provided for large bitstrings")

def parse_lattice_timeseries(filename):
    """Parse entire timeseries file into lattice data."""
    try:
        # Read the data
        data = pd.read_csv(filename, sep='\t', header=None)
        
        # First column is time step, second column contains lattice states
        time_steps = data.iloc[:, 0].values
        lattice_strings = data.iloc[:, 1].values
        
        # Parse lattice states
        lattice_matrix = []
        B = None  # Will be determined from first entry
        
        for lattice_str in lattice_strings:
            # Split by comma to get individual agent languages
            agents = lattice_str.split(',')
            
            # Store each agent's language as a tuple of bits
            lattice_row = []
            for agent_lang in agents:
                bits = tuple(int(b) for b in agent_lang)
                lattice_row.append(bits)
            
            if B is None:
                B = len(agents[0])  # Bits per agent
            
            lattice_matrix.append(lattice_row)
        
        L = len(agents)  # Number of agents
        
        return time_steps, lattice_matrix, L, B
        
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None, None, None

def get_last_line_from_file(filename):
    """Get the last non-empty line from a file."""
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return None
        f.seek(-1, 2)
        lines = []
        while f.tell() > 0:
            char = f.read(1)
            if char == b'\n':
                if lines:
                    break
            f.seek(-2, 1)
            lines.append(char)
        if f.tell() == 0:
            f.seek(0)
            remaining = f.read().decode('utf-8')
            return remaining.strip().split('\n')[-1]
        last_line = b''.join(reversed(lines)).decode('utf-8').strip()
        return last_line if last_line else None

def parse_lattice_line(line):
    """Parse a single line into a 1D lattice array."""
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    
    step = int(parts[0])  # First column is the step number
    lattice_data = parts[1]  # Second column contains the lattice
    
    # Parse the lattice data: agents separated by ','
    agents = lattice_data.split(',')
    lattice = []
    for agent in agents:
        # Each agent is a bitstring, convert to tuple of ints
        bits = tuple(int(b) for b in agent)
        lattice.append(bits)
    
    return step, lattice

def load_all_timeseries(L, B, mu):
    """Load complete timeseries data from each file matching the specified parameters."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscan/L_{L}_B_{B}/g_*_a_*_mu_{mu}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files matching L={L}, B={B}, mu={mu}")
    
    timeseries_data = {}
    
    for filename in tqdm(files, desc="Loading timeseries data"):
        gamma, alpha, L_file, B_file, mu_file = extract_params_from_filename(filename)
        
        if gamma is None or alpha is None:
            continue
            
        # Double-check that the extracted parameters match what we're looking for
        if L_file != L or B_file != B or mu_file != mu:
            continue
            
        try:
            time_steps, lattice_matrix, L_current, B_current = parse_lattice_timeseries(filename)
            if lattice_matrix is not None:
                timeseries_data[(gamma, alpha)] = {
                    'time_steps': time_steps,
                    'lattice_matrix': lattice_matrix,
                    'L': L_current,
                    'B': B_current,
                    'mu': mu_file
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return timeseries_data

def create_lattice_timeseries_grid(timeseries_data, L, B, mu):
    """Create a grid plot of lattice timeseries organized by gamma and alpha."""
    if not timeseries_data:
        print("No timeseries data to plot")
        return
    
    # Get unique gamma and alpha values
    gammas = sorted(set(key[0] for key in timeseries_data.keys()))
    alphas = sorted(set(key[1] for key in timeseries_data.keys()), reverse=True)  # Reverse alpha order
    
    print(f"Gamma values: {gammas}")
    print(f"Alpha values: {alphas}")
    
    # Create color map for bitstrings
    color_map = None
    rainbow_colors = None
    if B > 4 and B <= 16:
        color_map = create_bitstring_color_map(B)
    elif B > 16:
        rainbow_colors = get_rainbow_color_list(n_colors=65536, seed=42)
    
    # Create the figure with gridspec for better control
    fig = plt.figure(figsize=(4*len(gammas), 3*len(alphas)))
    gs = gridspec.GridSpec(len(alphas), len(gammas), 
                          hspace=0.3, wspace=0.2)
    
    # Plot each timeseries
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            ax = fig.add_subplot(gs[i, j])
            
            if (gamma, alpha) in timeseries_data:
                data = timeseries_data[(gamma, alpha)]
                time_steps = data['time_steps']
                lattice_matrix = data['lattice_matrix']
                L_current = data['L']
                B_current = data['B']
                
                # Convert lattice data to RGB image (like in viz_lattice_timeseries.py)
                n_timesteps = len(lattice_matrix)
                rgb_image = np.zeros((n_timesteps, L_current, 3))
                
                for t, lattice_row in enumerate(lattice_matrix):
                    for agent_idx, bitstring in enumerate(lattice_row):
                        color = bitstring_to_color(bitstring, color_map=color_map, rainbow_colors=rainbow_colors)
                        rgb_image[t, agent_idx] = color
                
                # Plot the space-time diagram
                ax.imshow(rgb_image, aspect='auto', interpolation='nearest', origin='lower')
                ax.set_title(f'γ={gamma}, α={alpha}', fontsize=10)
                
                # Set axis labels for edge plots only
                if i == len(alphas) - 1:  # Bottom row
                    ax.set_xlabel('Lattice Position', fontsize=8)
                if j == 0:  # Left column
                    ax.set_ylabel('Time Step', fontsize=8)
                
                # Reduce tick density
                n_x_ticks = min(5, L_current//100) if L_current > 100 else L_current//10
                n_y_ticks = min(5, n_timesteps//100) if n_timesteps > 100 else n_timesteps//10
                
                if n_x_ticks > 0:
                    ax.set_xticks(range(0, L_current, max(1, L_current//n_x_ticks)))
                if n_y_ticks > 0:
                    ax.set_yticks(range(0, n_timesteps, max(1, n_timesteps//n_y_ticks)))
                    ax.set_yticklabels([time_steps[idx] for idx in range(0, n_timesteps, max(1, n_timesteps//n_y_ticks))])
                
                # Make tick labels smaller
                ax.tick_params(axis='both', which='major', labelsize=6)
                
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'γ={gamma}, α={alpha}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall title and labels
    fig.suptitle(f'1D Lattice Evolution Grid (Colored by Language)\n(L={L}, B={B}, μ={mu})', 
                fontsize=16, y=0.98)
    
    # Add axis labels to the entire figure
    fig.text(0.5, 0.02, 'Gamma (Global Interaction Strength)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', rotation='vertical', fontsize=12)
    
    # Save the plot with parameters in filename
    output_dir = os.path.join(os.path.dirname(__file__), "plots/latticeGrid")
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/timeseries_grid_L_{L}_B_{B}_mu_{mu}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')  # Reduced DPI for large plots
    plt.close()  # Close to free memory
    
    print(f"Plot saved to: {fname}")

def plot_raster(L=1024, B=16, mu=0.001):
    """Plot alpha-gamma raster (timeseries grid) with fixed mu."""
    print(f"Looking for files with parameters: L={L}, B={B}, mu={mu}")
    
    # Load complete timeseries from files matching the specified parameters
    timeseries_data = load_all_timeseries(L, B, mu)
    
    if not timeseries_data:
        print("No valid timeseries data found.")
        return
    
    print(f"Successfully loaded {len(timeseries_data)} timeseries")
    
    # Create the grid plot
    create_lattice_timeseries_grid(timeseries_data, L, B, mu)

def load_all_timeseries_mu_raster(L, B, gamma):
    """Load complete timeseries data from each file matching the specified parameters from rasterscanMu folder."""
    pattern = os.path.join(os.path.dirname(__file__), f"outputs/latticeTimeseries/rasterscanMu/L_{L}_B_{B}/g_{gamma}_a_*_mu_*.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found with pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files in rasterscanMu/L_{L}_B_{B} folder matching gamma={gamma}")
    
    timeseries_data = {}
    
    for filename in tqdm(files, desc="Loading timeseries data"):
        gamma_file, alpha, L_file, B_file, mu = extract_params_from_filename(filename)
        
        if gamma_file is None or alpha is None or mu is None:
            continue
            
        if gamma_file != gamma or L_file != L or B_file != B:
            continue
            
        try:
            time_steps, lattice_matrix, L_current, B_current = parse_lattice_timeseries(filename)
            if lattice_matrix is not None:
                timeseries_data[(alpha, mu)] = {
                    'time_steps': time_steps,
                    'lattice_matrix': lattice_matrix,
                    'L': L_current,
                    'B': B_current,
                    'gamma': gamma_file,
                    'alpha': alpha,
                    'mu': mu
                }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return timeseries_data

def create_lattice_mu_timeseries_grid(timeseries_data, L, B, gamma):
    """Create a grid plot of lattice timeseries organized by mu (x-axis) and alpha (y-axis)."""
    if not timeseries_data:
        print("No timeseries data to plot")
        return
    
    # Get unique alpha and mu values
    alphas = sorted(set(key[0] for key in timeseries_data.keys()), reverse=True)  # Reverse alpha order
    mus = sorted(set(key[1] for key in timeseries_data.keys()))
    
    print(f"Alpha values: {alphas}")
    print(f"Mu values: {mus}")
    
    # Create color map for bitstrings
    color_map = None
    rainbow_colors = None
    if B > 4 and B <= 16:
        color_map = create_bitstring_color_map(B)
    elif B > 16:
        rainbow_colors = get_rainbow_color_list(n_colors=65536, seed=42)
    
    # Create the figure with gridspec for better control
    fig = plt.figure(figsize=(4*len(mus), 3*len(alphas)))
    gs = gridspec.GridSpec(len(alphas), len(mus), 
                          hspace=0.3, wspace=0.2)
    
    # Plot each timeseries
    for i, alpha in enumerate(alphas):
        for j, mu in enumerate(mus):
            ax = fig.add_subplot(gs[i, j])
            
            if (alpha, mu) in timeseries_data:
                data = timeseries_data[(alpha, mu)]
                time_steps = data['time_steps']
                lattice_matrix = data['lattice_matrix']
                L_current = data['L']
                B_current = data['B']
                
                # Convert lattice data to RGB image (like in viz_lattice_timeseries.py)
                n_timesteps = len(lattice_matrix)
                rgb_image = np.zeros((n_timesteps, L_current, 3))
                
                for t, lattice_row in enumerate(lattice_matrix):
                    for agent_idx, bitstring in enumerate(lattice_row):
                        color = bitstring_to_color(bitstring, color_map=color_map, rainbow_colors=rainbow_colors)
                        rgb_image[t, agent_idx] = color
                
                # Plot the space-time diagram
                ax.imshow(rgb_image, aspect='auto', interpolation='nearest', origin='lower')
                ax.set_title(f'α={alpha}, μ={mu:.4f}', fontsize=10)
                
                # Set axis labels for edge plots only
                if i == len(alphas) - 1:  # Bottom row
                    ax.set_xlabel('Lattice Position', fontsize=8)
                if j == 0:  # Left column
                    ax.set_ylabel('Time Step', fontsize=8)
                
                # Reduce tick density
                n_x_ticks = min(5, L_current//100) if L_current > 100 else L_current//10
                n_y_ticks = min(5, n_timesteps//100) if n_timesteps > 100 else n_timesteps//10
                
                if n_x_ticks > 0:
                    ax.set_xticks(range(0, L_current, max(1, L_current//n_x_ticks)))
                if n_y_ticks > 0:
                    ax.set_yticks(range(0, n_timesteps, max(1, n_timesteps//n_y_ticks)))
                    ax.set_yticklabels([time_steps[idx] for idx in range(0, n_timesteps, max(1, n_timesteps//n_y_ticks))])
                
                # Make tick labels smaller
                ax.tick_params(axis='both', which='major', labelsize=6)
                
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'α={alpha}, μ={mu:.4f}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall title and labels
    fig.suptitle(f'1D Lattice Evolution Grid: Alpha vs Mu (Colored by Language)\n(L={L}, B={B}, γ={gamma})', 
                fontsize=16, y=0.98)
    
    # Add axis labels to the entire figure
    fig.text(0.5, 0.02, 'Mu (Mutation Rate)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Alpha (Local Interaction Strength)', va='center', rotation='vertical', fontsize=12)
    
    # Save the plot with parameters in filename
    output_dir = os.path.join(os.path.dirname(__file__), "plots/latticeGridMu")
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/timeseries_grid_L_{L}_B_{B}_gamma_{gamma}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')  # Reduced DPI for large plots
    plt.close()  # Close to free memory
    
    print(f"Plot saved to: {fname}")

def plot_raster_mu(L=1024, B=16, gamma=1):
    """Plot alpha-mu raster (timeseries grid) with fixed gamma."""
    print(f"Looking for files with parameters: L={L}, B={B}, gamma={gamma} in rasterscanMu folder")
    
    # Load complete timeseries from files matching the specified parameters
    timeseries_data = load_all_timeseries_mu_raster(L, B, gamma)
    
    if not timeseries_data:
        print("No valid timeseries data found.")
        return
    
    print(f"Successfully loaded {len(timeseries_data)} timeseries")
    
    # Create the grid plot
    create_lattice_mu_timeseries_grid(timeseries_data, L, B, gamma)

if __name__ == "__main__":
    # plot_raster(L=1024, B=16, mu=0.001)
    plot_raster_mu(L=1024, B=16, gamma=1)