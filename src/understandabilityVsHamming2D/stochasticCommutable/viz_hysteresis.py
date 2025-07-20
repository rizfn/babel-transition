import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from collections import defaultdict


def parse_filename(filename):
    """Extract parameters from hysteresis filename."""
    # Example filename: hysteresis_all_ones_g_1.000000_a_1.200000_mu_0.000100_1752947536.tsv
    # or: hysteresis_all_zeros_g_1.000000_a_0.900000_mu_0.000100_1752947536.tsv
    basename = os.path.basename(filename)
    
    # Extract initial condition type
    if 'all_ones' in basename:
        init_condition = 'all_ones'
    elif 'all_zeros' in basename:
        init_condition = 'all_zeros'
    elif 'loaded_state' in basename:  # Keep for backward compatibility
        init_condition = 'loaded_state'
    else:
        return None
    
    # Extract parameters using regex
    g_match = re.search(r'g_([\d.]+)', basename)
    a_match = re.search(r'a_([\d.]+)', basename)
    mu_match = re.search(r'mu_([\d.]+)', basename)
    
    if not (g_match and a_match and mu_match):
        return None
    
    return {
        'gamma': float(g_match.group(1)),
        'alpha': float(a_match.group(1)),
        'mu': float(mu_match.group(1)),
        'init_condition': init_condition
    }

def parse_lattice_final_state(filepath, B):
    """Parse the final state from a hysteresis file and count all-ones bitstrings."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Get the last non-empty line
        last_line = None
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break
        
        if not last_line:
            return None
        
        # Parse the lattice data
        step, lattice_str = last_line.split('\t')
        rows = lattice_str.split(';')
        
        all_ones_count = 0
        total_agents = 0
        all_ones_bitstring = '1' * B
        
        for row in rows:
            cells = row.split(',')
            for cell in cells:
                if len(cell) == B:
                    total_agents += 1
                    if cell == all_ones_bitstring:
                        all_ones_count += 1
        
        if total_agents == 0:
            return None
        
        fraction = all_ones_count / total_agents
        return fraction
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def collect_hysteresis_data(hysteresis_dir, B=16):
    """Collect all hysteresis data from files."""
    data = defaultdict(lambda: defaultdict(list))  # data[init_condition][alpha] = [fractions]
    
    # Find all hysteresis files
    pattern = os.path.join(hysteresis_dir, "**", "hysteresis_*.tsv")
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} hysteresis files")
    
    for filepath in files:
        params = parse_filename(filepath)
        if params is None:
            continue
        
        fraction = parse_lattice_final_state(filepath, B)
        if fraction is None:
            continue
        
        alpha = params['alpha']
        init_condition = params['init_condition']
        
        data[init_condition][alpha].append(fraction)
        print(f"Processed: {os.path.basename(filepath)} -> α={alpha}, init={init_condition}, fraction={fraction:.4f}")
    
    return data


def plot_hysteresis(data, output_path=None):
    """Plot hysteresis data with means and standard deviations."""
    plt.figure(figsize=(10, 6))
    
    colors = {'all_ones': 'blue', 'all_zeros': 'red', 'loaded_state': 'red'}  # Keep loaded_state for backward compatibility
    labels = {'all_ones': 'All-ones initial condition', 'all_zeros': 'All-zeros initial condition', 'loaded_state': 'Loaded initial condition'}
    
    for init_condition in ['all_ones', 'all_zeros', 'loaded_state']:
        if init_condition not in data:
            continue
        
        alphas = sorted(data[init_condition].keys())
        means = []
        stds = []
        
        for alpha in alphas:
            fractions = data[init_condition][alpha]
            if fractions:
                means.append(np.mean(fractions))
                stds.append(np.std(fractions))
            else:
                means.append(0)
                stds.append(0)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot mean line
        plt.plot(alphas, means, 'o-', color=colors[init_condition], 
                label=labels[init_condition], linewidth=2, markersize=6)
        
        # Shade standard deviation
        plt.fill_between(alphas, means - stds, means + stds, 
                        color=colors[init_condition], alpha=0.3)
    
    plt.xlabel('Alpha (α)', fontsize=12)
    plt.ylabel('Fraction of all-ones bitstrings', fontsize=12)
    plt.title('Hysteresis: Final state composition vs Alpha', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True)    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

def parse_timeseries_data(filepath, B):
    """Parse timeseries data and count populations for each timestep."""
    timeseries_data = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                # Parse the lattice data
                step, lattice_str = line.strip().split('\t')
                rows = lattice_str.split(';')
                
                # Count bitstring populations
                bitstring_counts = defaultdict(int)
                total_agents = 0
                
                for row in rows:
                    cells = row.split(',')
                    for cell in cells:
                        if len(cell) == B:
                            bitstring_counts[cell] += 1
                            total_agents += 1
                
                # Convert to fractions
                bitstring_fractions = {bs: count/total_agents for bs, count in bitstring_counts.items()}
                
                timeseries_data.append({
                    'step': int(step),
                    'populations': bitstring_fractions
                })
                
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
    
    return timeseries_data

def get_top_bitstrings(all_timeseries_data, top_n=20):
    """Find the top N bitstrings by maximum population across all timepoints and files."""
    max_populations = defaultdict(float)
    
    for file_data in all_timeseries_data:
        for timepoint in file_data:
            for bitstring, fraction in timepoint['populations'].items():
                max_populations[bitstring] = max(max_populations[bitstring], fraction)
    
    # Sort by maximum population and take top N
    sorted_bitstrings = sorted(max_populations.items(), key=lambda x: x[1], reverse=True)
    return [bs for bs, _ in sorted_bitstrings[:top_n]]

def plot_timeseries(alpha, mu, B=16, top_n=20):
    """Plot time series of top N population counts for given parameters."""
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hysteresis_dir = os.path.join(base_dir, "outputs", "hysteresis")
    plots_dir = os.path.join(base_dir, "plots/hysteresis")
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert parameters to the 6-decimal format used in filenames
    alpha_str = f"{alpha:.6f}"
    mu_str = f"{mu:.6f}"
    
    # Find all matching files using the full decimal format
    pattern = os.path.join(hysteresis_dir, "**", f"*_a_{alpha_str}_mu_{mu_str}_*.tsv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No files found matching pattern: *_a_{alpha_str}_mu_{mu_str}_*.tsv")
        print(f"Looking in directory: {hysteresis_dir}")
        # Debug: show what files are actually there
        all_files = glob.glob(os.path.join(hysteresis_dir, "**", "*.tsv"), recursive=True)
        print(f"Found {len(all_files)} total .tsv files")
        if all_files:
            print("Sample filenames:")
            for i, f in enumerate(all_files[:3]):
                print(f"  {os.path.basename(f)}")
        return
    
    print(f"Found {len(files)} files for α={alpha}, μ={mu}")
    
    # Parse all timeseries data and organize by initial condition
    timeseries_by_condition = {'all_ones': [], 'all_zeros': []}
    
    for filepath in files:
        params = parse_filename(filepath)
        if params is None:
            continue
        
        timeseries_data = parse_timeseries_data(filepath, B)
        if timeseries_data is None:
            continue
        
        init_condition = params['init_condition']
        if init_condition in timeseries_by_condition:
            timeseries_by_condition[init_condition].append({
                'data': timeseries_data,
                'filename': os.path.basename(filepath)
            })
            print(f"Processed: {os.path.basename(filepath)} -> {init_condition}")
    
    # Determine the grid size
    max_files = max(len(timeseries_by_condition['all_ones']), len(timeseries_by_condition['all_zeros']))
    if max_files == 0:
        print("No valid timeseries data found!")
        return
    
    # Find top bitstrings across all data
    all_timeseries_data = []
    for condition_data in timeseries_by_condition.values():
        for file_data in condition_data:
            all_timeseries_data.append(file_data['data'])
    
    top_bitstrings = get_top_bitstrings(all_timeseries_data, top_n)
    
    # Create color map for bitstrings
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_bitstrings)))
    bitstring_colors = dict(zip(top_bitstrings, colors))
    
    # Plot setup - rows = max files, cols = 2 (all_ones, all_zeros)
    fig, axes = plt.subplots(max_files, 2, figsize=(12, 4*max_files), sharex=True, sharey=True)
    
    # Handle case where there's only one row
    if max_files == 1:
        axes = axes.reshape(1, -1)
    
    condition_labels = {
        'all_ones': 'All-ones initial condition',
        'all_zeros': 'All-zeros initial condition'
    }
    
    # Plot data
    for col_idx, (condition, condition_files) in enumerate([('all_ones', timeseries_by_condition['all_ones']), 
                                                           ('all_zeros', timeseries_by_condition['all_zeros'])]):
        
        for row_idx in range(max_files):
            ax = axes[row_idx, col_idx]
            
            if row_idx < len(condition_files):
                # We have data for this position
                file_info = condition_files[row_idx]
                timeseries_data = file_info['data']
                filename = file_info['filename']
                
                # Organize data by bitstring
                bitstring_timeseries = defaultdict(list)
                steps = []
                
                for timepoint in timeseries_data:
                    steps.append(timepoint['step'])
                    for bitstring in top_bitstrings:
                        fraction = timepoint['populations'].get(bitstring, 0.0)
                        bitstring_timeseries[bitstring].append(fraction)
                
                # Plot each bitstring timeseries
                plotted_bitstrings = []
                for bitstring in top_bitstrings:
                    if any(f > 0.01 for f in bitstring_timeseries[bitstring]):  # Only plot if it reaches >1% at some point
                        ax.plot(steps, bitstring_timeseries[bitstring], 
                               color=bitstring_colors[bitstring], 
                               label=bitstring, 
                               linewidth=1.5, 
                               alpha=0.8)
                        plotted_bitstrings.append(bitstring)
                
                # Set title for this subplot
                if row_idx == 0:
                    ax.set_title(f'{condition_labels[condition]}', fontsize=12)
                
                # Add filename as text in the plot
                ax.text(0.02, 0.98, filename, transform=ax.transAxes, fontsize=8, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add legend inside each subplot
                if plotted_bitstrings:
                    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85), framealpha=0.8)
                
            else:
                # No data for this position, hide the subplot
                ax.set_visible(False)
                continue
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
    plt.suptitle(f'Top {top_n} Bitstring Populations vs Time (α={alpha}, μ={mu})', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(plots_dir, f"timeseries_alpha_{alpha}_mu_{mu}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def main():
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hysteresis_dir = os.path.join(base_dir, "outputs", "hysteresis")
    plots_dir = os.path.join(base_dir, "plots/hysteresis")
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if hysteresis directory exists
    if not os.path.exists(hysteresis_dir):
        print(f"Hysteresis directory not found: {hysteresis_dir}")
        print("Make sure to run the hysteresis simulations first.")
        return
    
    print(f"Looking for hysteresis data in: {hysteresis_dir}")
    
    # Collect data
    data = collect_hysteresis_data(hysteresis_dir, B=16)
    
    if not data:
        print("No hysteresis data found!")
        return
    
    # Create plot
    output_path = os.path.join(plots_dir, "hysteresis_analysis.png")
    plot_hysteresis(data, output_path)

if __name__ == "__main__":
    # main()  # for the hysteresis plot
    plot_timeseries(alpha=1.0, mu=0.0001)