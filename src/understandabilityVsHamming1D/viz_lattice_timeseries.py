import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from itertools import product

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

def bitstring_to_color(bits, color_map=None):
    """Map a bitstring to a unique RGB color."""
    if len(bits) == 4:
        r = 0.3 + 0.7 * bits[0]
        g = 0.3 + 0.7 * bits[1]
        b = 0.3 + 0.7 * bits[2]
        brightness = 0.3 + 0.5 * bits[3]
        color = np.array([r, g, b]) * brightness
        color = np.clip(color, 0, 1)
    elif len(bits) == 2:
        base = 0.2
        delta = 0.6
        r = base + delta * bits[0]
        g = base + delta * bits[1]
        b = base + delta * (bits[0] ^ bits[1])  # XOR for more separation
        return [r, g, b]
    else:
        # For longer bitstrings, use the provided color map
        if color_map is None:
            raise ValueError("color_map must be provided for bitstrings longer than 4 bits")
        
        bitstring_tuple = tuple(bits)
        if bitstring_tuple not in color_map:
            raise ValueError(f"Bitstring {bitstring_tuple} not found in color_map")
        
        return color_map[bitstring_tuple]
    
    return color

def load_lattice_data(filepath):
    """Load lattice timeseries data from TSV file."""
    try:
        # Read the data
        data = pd.read_csv(filepath, sep='\t', header=None)
        
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
        print(f"Error loading data from {filepath}: {e}")
        return None, None, None, None

def plot_lattice_timeseries(time_steps, lattice_matrix, L, B, gamma, alpha, mu, output_path):
    """Create and save the lattice timeseries visualization."""
    
    # Create color map for bitstrings
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    # Convert lattice data to RGB image
    n_timesteps = len(lattice_matrix)
    rgb_image = np.zeros((n_timesteps, L, 3))
    
    for t, lattice_row in enumerate(lattice_matrix):
        for i, bitstring in enumerate(lattice_row):
            color = bitstring_to_color(bitstring, color_map)
            rgb_image[t, i] = color
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot the space-time diagram
    ax.imshow(rgb_image, aspect='auto', interpolation='nearest', origin='lower')
    ax.set_xlabel('Lattice Position')
    ax.set_ylabel('Time Step')
    ax.set_title(f'1D Lattice Evolution (L={L}, B={B})\nγ={gamma}, α={alpha}, μ={mu}')
    
    # Set axis labels
    ax.set_xticks(range(0, L, max(1, L//10)))
    ax.set_yticks(range(0, n_timesteps, max(1, n_timesteps//10)))
    ax.set_yticklabels([time_steps[i] for i in range(0, n_timesteps, max(1, n_timesteps//10))])
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize lattice timeseries data')
    parser.add_argument('--L', type=int, default=1024, help='Lattice size')
    parser.add_argument('--B', type=int, default=16, help='Bits per agent')
    parser.add_argument('--gamma', type=float, default=1, help='Global interaction strength')
    parser.add_argument('--alpha', type=float, default=1.2, help='Local interaction strength')
    parser.add_argument('--mu', type=float, default=4.64159e-5, help='Mutation rate')
    parser.add_argument('--input_dir', type=str, default=None, 
                       help='Input directory (if not specified, uses default structure)')
    
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Construct input file path
    if args.input_dir:
        input_path = Path(args.input_dir) / f"g_{args.gamma}_a_{args.alpha}_mu_{args.mu}.tsv"
    else:
        input_path = (script_dir / "outputs" / "latticeTimeseries" / "long" / 
                     f"L_{args.L}_B_{args.B}" / f"g_{args.gamma}_a_{args.alpha}_mu_{args.mu}.tsv")
    
    # Construct output file path
    output_dir = (script_dir / "plots" / "latticeTimeseries" / 
                  f"L_{args.L}_B_{args.B}")
    output_path = output_dir / f"g_{args.gamma}_a_{args.alpha}_mu_{args.mu}.png"
    
    print(f"Loading data from: {input_path}")
    
    # Load data
    time_steps, lattice_matrix, L, B = load_lattice_data(input_path)
    
    if lattice_matrix is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded data: {len(time_steps)} time steps, {L} agents, {B} bits per agent")
    
    # Create visualization
    plot_lattice_timeseries(time_steps, lattice_matrix, L, B, 
                           args.gamma, args.alpha, args.mu, str(output_path))

if __name__ == "__main__":
    main()