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

def communicability(a, b, B):
    """Calculate communicability between two language bitstrings."""
    count = 0
    for i in range(B):
        count += (a[i] & b[i])
    return count

def mean_field_distance(language, mean_field, B):
    """Calculate mean-field distance (equivalent to hamming distance with mean field)."""
    distance = 0.0
    for i in range(B):
        distance += abs(float(language[i]) - mean_field[i])
    return distance / float(B)

def calculate_fitness_components(lattice_matrix, gamma, alpha, L, B):
    """Calculate fitness components for each agent at each time step."""
    n_timesteps = len(lattice_matrix)
    
    # Initialize fitness arrays
    global_fitness = np.zeros((n_timesteps, L))
    local_fitness = np.zeros((n_timesteps, L))
    total_fitness = np.zeros((n_timesteps, L))
    
    for t in range(n_timesteps):
        lattice_row = lattice_matrix[t]
        
        # 1. Calculate mean-field bitstring for this timestep
        mean_field = np.zeros(B)
        for agent_lang in lattice_row:
            for b in range(B):
                mean_field[b] += float(agent_lang[b])
        mean_field /= float(L)  # Normalize to get probabilities
        
        # 2. Calculate fitness for each agent
        for i in range(L):
            agent_lang = lattice_row[i]
            
            # 2a. Global fitness component
            global_fit = gamma * mean_field_distance(agent_lang, mean_field, B)
            global_fitness[t, i] = global_fit
            
            # 2b. Local fitness component
            left_neighbor = lattice_row[(i - 1 + L) % L]
            right_neighbor = lattice_row[(i + 1) % L]
            
            comm_left = communicability(agent_lang, left_neighbor, B)
            comm_right = communicability(agent_lang, right_neighbor, B)
            
            local_fit = (alpha / 2.0) * ((comm_left + comm_right) / float(B))
            local_fitness[t, i] = local_fit
            
            # Total fitness
            total_fitness[t, i] = global_fit + local_fit
    
    return global_fitness, local_fitness, total_fitness

def calculate_fitness_comparison_colors(total_fitness, L):
    """Calculate colors based on fitness comparison with neighbors."""
    n_timesteps, _ = total_fitness.shape
    
    # Initialize color array (RGB)
    fitness_colors = np.zeros((n_timesteps, L, 3))
    
    for t in range(n_timesteps):
        for i in range(L):
            left_neighbor_idx = (i - 1 + L) % L
            right_neighbor_idx = (i + 1) % L
            
            current_fitness = total_fitness[t, i]
            left_fitness = total_fitness[t, left_neighbor_idx]
            right_fitness = total_fitness[t, right_neighbor_idx]
            
            # Compare with neighbors
            higher_than_left = current_fitness > left_fitness
            higher_than_right = current_fitness > right_fitness
            
            if higher_than_left and higher_than_right:
                # White: higher than both neighbors
                fitness_colors[t, i] = [1.0, 1.0, 1.0]
            elif not higher_than_left and not higher_than_right:
                # Black: lower than both neighbors
                fitness_colors[t, i] = [0.0, 0.0, 0.0]
            elif higher_than_left and not higher_than_right:
                # Red: higher than left, lower than right
                fitness_colors[t, i] = [1.0, 0.0, 0.0]
            elif not higher_than_left and higher_than_right:
                # Blue: higher than right, lower than left
                fitness_colors[t, i] = [0.0, 0.0, 1.0]
    
    return fitness_colors

def create_spatial_language_heatmap(lattice_matrix, L, B, color_map=None):
    """Create a heatmap showing languages of all agents in their spatial arrangement from the last timestep."""
    # Use the last timestep
    last_timestep = lattice_matrix[-1]
    
    # Check if this is 2D data (contains ';' separators)
    if isinstance(last_timestep[0], str):
        # This is 2D data - parse it
        # For 2D lattice data, we need to parse the string format
        # Format should be: "agent1,agent2,...;agent1,agent2,...;..."
        lattice_2d = []
        for row_str in last_timestep:
            # Split by ';' to get rows, then by ',' to get agents
            rows = row_str.split(';')
            lattice_rows = []
            for row in rows:
                agents = row.split(',')
                agent_row = []
                for agent_lang in agents:
                    bits = tuple(int(b) for b in agent_lang)
                    agent_row.append(bits)
                lattice_rows.append(agent_row)
            lattice_2d.extend(lattice_rows)
        
        # Assume square lattice
        lattice_size = int(len(lattice_2d) ** 0.5)
        
        # Create heatmap: each position (i,j) shows bits for that agent
        heatmap = np.ones((lattice_size * B, lattice_size, 3))  # Start with white
        
        for i in range(lattice_size):
            for j in range(lattice_size):
                if i < len(lattice_2d) and j < len(lattice_2d[i]):
                    language = lattice_2d[i][j]
                    color = bitstring_to_color(language, color_map)
                    
                    # Fill bits for this agent
                    for bit_idx, bit_value in enumerate(language):
                        row_idx = i * B + bit_idx
                        if bit_value == 1:
                            # Bit is on: use language color
                            heatmap[row_idx, j] = color
                        else:
                            # Bit is off: use white
                            heatmap[row_idx, j] = [1.0, 1.0, 1.0]
        
        return heatmap, lattice_size, lattice_2d
    
    else:
        # This is 1D data - treat as a single row
        # Create heatmap: each column is an agent, each row is a bit
        heatmap = np.ones((B, L, 3))  # Start with white
        
        for i in range(L):
            language = last_timestep[i]
            color = bitstring_to_color(language, color_map)
            
            for bit_idx, bit_value in enumerate(language):
                if bit_value == 1:
                    # Bit is on: use language color
                    heatmap[bit_idx, i] = color
                else:
                    # Bit is off: use white
                    heatmap[bit_idx, i] = [1.0, 1.0, 1.0]
        
        return heatmap, L, last_timestep

def load_lattice_data(filepath):
    """Load lattice timeseries data from TSV file."""
    try:
        # Read the data
        data = pd.read_csv(filepath, sep='\t', header=None)
        
        # First column is time step, second column contains lattice states
        time_steps = data.iloc[:, 0].values
        lattice_strings = data.iloc[:, 1].values
        
        # Check if this is 2D data (contains ';' separators indicating 2D structure)
        if ';' in lattice_strings[0]:
            # This is 2D data - parse differently
            lattice_matrix = []
            B = None
            L = None
            
            for lattice_str in lattice_strings:
                # Split by ';' to get rows
                rows = lattice_str.split(';')
                if L is None:
                    L = int(len(rows) ** 0.5)  # Assume square lattice
                
                # Parse each row
                lattice_timestep = []
                for row in rows:
                    agents = row.split(',')
                    for agent_lang in agents:
                        bits = tuple(int(b) for b in agent_lang)
                        lattice_timestep.append(bits)
                        if B is None:
                            B = len(agent_lang)
                
                lattice_matrix.append(lattice_timestep)
            
            return time_steps, lattice_matrix, L*L, B
        
        else:
            # This is 1D data - parse as before
            lattice_matrix = []
            B = None
            
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

def plot_lattice_and_fitness(time_steps, lattice_matrix, L, B, gamma, alpha, mu, output_path):
    """Create and save the lattice timeseries and fitness visualizations."""
    
    print("Calculating fitness components...")
    global_fitness, local_fitness, total_fitness = calculate_fitness_components(
        lattice_matrix, gamma, alpha, L, B)
    
    print("Calculating fitness comparison colors...")
    fitness_comparison_colors = calculate_fitness_comparison_colors(total_fitness, L)
    
    # Create color map for bitstrings
    color_map = None
    if B > 4:
        color_map = create_bitstring_color_map(B)
    
    # Convert lattice data to RGB image
    n_timesteps = len(lattice_matrix)
    rgb_image = np.zeros((n_timesteps, L, 3))
    
    print("Converting lattice to colors...")
    for t, lattice_row in enumerate(lattice_matrix):
        for i, bitstring in enumerate(lattice_row):
            color = bitstring_to_color(bitstring, color_map)
            rgb_image[t, i] = color
    
    print("Creating spatial language heatmap...")
    spatial_heatmap, spatial_size, spatial_data = create_spatial_language_heatmap(lattice_matrix, L, B, color_map)
    
    # Create the plot with 6 subplots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    # Plot 1: Lattice evolution (colored by language)
    ax1 = axes[0, 0]
    ax1.imshow(rgb_image, aspect='auto', interpolation='nearest', origin='lower')
    ax1.set_xlabel('Lattice Position')
    ax1.set_ylabel('Time Step')
    ax1.set_title('Language Evolution')
    
    # Plot 2: Global fitness component
    ax2 = axes[0, 1]
    im2 = ax2.imshow(global_fitness, aspect='auto', interpolation='nearest', 
                     origin='lower', cmap='viridis')
    ax2.set_xlabel('Lattice Position')
    ax2.set_ylabel('Time Step')
    ax2.set_title(f'Global Fitness Component (γ={gamma})')
    plt.colorbar(im2, ax=ax2, label='Global Fitness')
    
    # Plot 3: Local fitness component
    ax3 = axes[0, 2]
    im3 = ax3.imshow(local_fitness, aspect='auto', interpolation='nearest', 
                     origin='lower', cmap='plasma')
    ax3.set_xlabel('Lattice Position')
    ax3.set_ylabel('Time Step')
    ax3.set_title(f'Local Fitness Component (α={alpha})')
    plt.colorbar(im3, ax=ax3, label='Local Fitness')
    
    # Plot 4: Total fitness
    ax4 = axes[1, 0]
    im4 = ax4.imshow(total_fitness, aspect='auto', interpolation='nearest', 
                     origin='lower', cmap='coolwarm')
    ax4.set_xlabel('Lattice Position')
    ax4.set_ylabel('Time Step')
    ax4.set_title('Total Fitness (Global + Local)')
    plt.colorbar(im4, ax=ax4, label='Total Fitness')
    
    # Plot 5: Fitness comparison with neighbors
    ax5 = axes[1, 1]
    ax5.imshow(fitness_comparison_colors, aspect='auto', interpolation='nearest', origin='lower')
    ax5.set_xlabel('Lattice Position')
    ax5.set_ylabel('Time Step')
    ax5.set_title('Fitness vs Neighbors\n(White=Higher than both, Black=Lower than both,\nRed=Higher left/Lower right, Blue=Higher right/Lower left)')
    
    # Plot 6: Spatial language domains heatmap
    ax6 = axes[1, 2]
    ax6.imshow(spatial_heatmap, aspect='auto', interpolation='nearest')
    
    # Determine if this is 2D or 1D data for labeling
    if ';' in str(lattice_matrix[0]) if isinstance(lattice_matrix[0], str) else len(spatial_data) != L:
        # 2D data
        ax6.set_xlabel('Spatial Position (X)')
        ax6.set_ylabel('Agent × Bit (Y × Bit)')
        ax6.set_title(f'Spatial Language Domains\n(Last timestep, {spatial_size}×{spatial_size} lattice)')
        
        # Add horizontal lines to separate different spatial Y positions
        for y in range(1, spatial_size):
            ax6.axhline(y=y * B - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    else:
        # 1D data
        ax6.set_xlabel('Agent Position')
        ax6.set_ylabel('Bit Position')
        ax6.set_title(f'Language Bit Structure\n(Last timestep, all {L} agents)')
        
        # Set bit position labels
        ax6.set_yticks(range(B))
        ax6.set_yticklabels([f'B{i}' for i in range(B)])
    
    # Set consistent axis ticks for other subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(range(0, L, max(1, L//10)))
        ax.set_yticks(range(0, n_timesteps, max(1, n_timesteps//10)))
        ax.set_yticklabels([time_steps[i] for i in range(0, n_timesteps, max(1, n_timesteps//10))])
    
    # Add overall title
    fig.suptitle(f'1D Lattice Evolution and Fitness Analysis\n(L={L}, B={B}, γ={gamma}, α={alpha}, μ={mu})', 
                 fontsize=18, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize lattice timeseries data with fitness analysis')
    parser.add_argument('--L', type=int, default=1024, help='Lattice size')
    parser.add_argument('--B', type=int, default=16, help='Bits per agent')
    parser.add_argument('--gamma', type=float, default=1, help='Global interaction strength')
    parser.add_argument('--alpha', type=float, default=0.4, help='Local interaction strength')
    parser.add_argument('--mu', type=float, default=1e-05, help='Mutation rate')
    parser.add_argument('--input_dir', type=str, default='src/understandabilityVsHamming1D/outputs/latticeTimeseries/rasterscanMu/L_1024_B_16', 
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
    output_path = output_dir / f"g_{args.gamma}_a_{args.alpha}_mu_{args.mu}_with_fitness.png"
    
    print(f"Loading data from: {input_path}")
    
    # Load data
    time_steps, lattice_matrix, L, B = load_lattice_data(input_path)
    
    if lattice_matrix is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded data: {len(time_steps)} time steps, {L} agents, {B} bits per agent")
    
    # Create visualization
    plot_lattice_and_fitness(time_steps, lattice_matrix, L, B, 
                            args.gamma, args.alpha, args.mu, str(output_path))

if __name__ == "__main__":
    main()