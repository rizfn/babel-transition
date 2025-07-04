import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_timeseries(gamma, alpha, N, B, mu, generations=None):
    """
    Plot fitness over time for given parameters.
    
    Parameters:
    gamma, alpha, N, B, mu: Model parameters
    generations: Optional, if provided will be used in title
    """
    # Construct filename
    base_dir = os.path.dirname(__file__)
    filename = os.path.join(base_dir, f"outputs/top50/fitness/g_{gamma}_a_{alpha}_N_{N}_B_{B}_mu_{mu}.tsv")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    # Read the data
    try:
        data = pd.read_csv(filename, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Validate columns
    required_columns = ['generation', 'max_fitness', 'avg_fitness']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: Missing required columns. Found: {list(data.columns)}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot both fitness traces
    ax.plot(data['generation'], data['max_fitness'], 'b-', linewidth=2, label='Max Fitness', alpha=0.8)
    ax.plot(data['generation'], data['avg_fitness'], 'r-', linewidth=2, label='Average Fitness', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    
    # Title with parameters
    title = f'Fitness Evolution Over Time\n'
    title += f'γ={gamma}, α={alpha}, N={N}, B={B}, μ={mu}'
    if generations is not None:
        title += f', Generations={generations}'
    ax.set_title(title, fontsize=14)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add some statistics to the plot
    final_max = data['max_fitness'].iloc[-1]
    final_avg = data['avg_fitness'].iloc[-1]
    max_max = data['max_fitness'].max()
    max_avg = data['avg_fitness'].max()
    
    # Add text box with statistics
    stats_text = f'Final Max: {final_max:.3f}\n'
    stats_text += f'Final Avg: {final_avg:.3f}\n'
    stats_text += f'Peak Max: {max_max:.3f}\n'
    stats_text += f'Peak Avg: {max_avg:.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(base_dir, "plots/fitness")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"{output_dir}/fitness_g_{gamma}_a_{alpha}_N_{N}_B_{B}_mu_{mu}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {plot_filename}")

def main():
    """
    Main function - edit parameters here
    """
    # Set your parameters here
    gamma = 1
    alpha = 2
    N = 1000
    B = 16
    mu = 0.01
    generations = 1000  # Optional, for title display
    
    # Create the plot
    plot_fitness_timeseries(gamma, alpha, N, B, mu, generations)

if __name__ == "__main__":
    main()