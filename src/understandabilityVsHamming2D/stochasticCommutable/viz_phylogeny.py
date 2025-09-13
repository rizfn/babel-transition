import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def load_phylogeny_data(tsv_path):
    """Load phylogeny data from TSV file"""
    df = pd.read_csv(tsv_path, sep='\t')
    return df

def analyze_population_dynamics(df):
    """Analyze population dynamics over time"""
    print("Population dynamics analysis:")
    print(f"Time range: {df['step'].min()} to {df['step'].max()}")
    print(f"Number of unique languages: {df['language_id'].nunique()}")
    print(f"Number of timesteps: {df['step'].nunique()}")
    
    # Show languages that persist the longest
    language_lifespans = df.groupby('language_id').agg({
        'step': ['min', 'max', 'count'],
        'largest_cluster_size': 'max'
    }).round(2)
    language_lifespans.columns = ['first_seen', 'last_seen', 'appearances', 'max_population']
    language_lifespans['lifespan'] = language_lifespans['last_seen'] - language_lifespans['first_seen']
    
    print("\nTop 10 longest-lived languages:")
    print(language_lifespans.nlargest(10, 'lifespan').to_string())
    
    return language_lifespans

def plot_population_over_time(df, output_dir, top_n=30):
    """Plot population dynamics over time for top languages"""
    # Get top languages by max population
    top_languages = df.groupby('language_id')['largest_cluster_size'].max().nlargest(top_n)
    
    plt.figure(figsize=(12, 8))
    
    for lang_id in top_languages.index:
        lang_data = df[df['language_id'] == lang_id].sort_values('step')
        plt.plot(lang_data['step'], lang_data['largest_cluster_size'], 
                label=f'Lang {lang_id}', marker='o', markersize=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Largest Cluster Size')
    plt.title(f'Population Dynamics - Top {top_n} Languages by Max Population')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir.replace('outputs', 'plots'), 'population_dynamics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Population dynamics plot saved to: {plot_path}")

def build_hierarchical_tree(df):
    """Build a phylogenetic tree with proper branching events"""
    
    # Get language birth information
    language_births = df.groupby('language_id').agg({
        'step': 'min',
        'parent_id': 'first'
    }).reset_index()
    language_births.rename(columns={'step': 'birth_step'}, inplace=True)
    
    # Get language death information (last time seen)
    language_deaths = df.groupby('language_id')['step'].max().reset_index()
    language_deaths.rename(columns={'step': 'death_step'}, inplace=True)
    
    # Merge birth and death info
    language_info = language_births.merge(language_deaths, on='language_id')
    
    # Clean parent_id data
    language_info['parent_id_clean'] = pd.to_numeric(language_info['parent_id'], errors='coerce')
    
    # Find initial languages (those present at the first timestep)
    first_step = df['step'].min()
    initial_languages = df[df['step'] == first_step]['language_id'].unique()
    
    print(f"Number of initial languages at step {first_step}: {len(initial_languages)}")

    # Create internal node counter for branching events
    internal_node_id = max(df['language_id'].max(), 1000) + 1
    
    # Track all nodes (languages + internal nodes)
    all_nodes = {}
    
    # Track which node currently represents each original language lineage
    language_lineage_map = {}
    
    # Add initial languages as children of root
    for lang_id in initial_languages:
        lang_data = language_info[language_info['language_id'] == lang_id].iloc[0]
        all_nodes[lang_id] = {
            'birth_step': lang_data['birth_step'],
            'death_step': lang_data['death_step'],
            'parent': 'ROOT',
            'children': [],
            'is_leaf': True
        }
        # Initially, each language represents its own lineage
        language_lineage_map[lang_id] = lang_id
    
    # Process branching events
    branching_events = []
    
    for _, lang_data in language_info.iterrows():
        lang_id = lang_data['language_id']
        parent_id = lang_data['parent_id_clean']
        birth_step = lang_data['birth_step']
        
        # Skip if this is an initial language or has no valid parent
        if lang_id in initial_languages or pd.isna(parent_id):
            continue
            
        # Check if parent exists in our data
        if parent_id not in language_info['language_id'].values:
            continue
            
        parent_data = language_info[language_info['language_id'] == parent_id].iloc[0]
        
        # Create branching event
        branching_events.append({
            'parent_id': int(parent_id),
            'child_id': int(lang_id),
            'branch_time': birth_step,
            'parent_birth': parent_data['birth_step'],
            'parent_death': parent_data['death_step'],
            'child_birth': birth_step,
            'child_death': lang_data['death_step']
        })
    # Sort branching events by time
    branching_events.sort(key=lambda x: x['branch_time'])
    
    # Process each branching event
    for event in branching_events:
        parent_id = event['parent_id']
        child_id = event['child_id']
        branch_time = event['branch_time']
        
        # Find the current node that represents the parent lineage
        current_parent_node = language_lineage_map.get(parent_id)
        
        if current_parent_node is None:
            # Find the most similar language from the previous step
            prev_step = branch_time - 500
            prev_step_languages = df[df['step'] == prev_step]['language_id'].unique()
            
            if len(prev_step_languages) == 0:
                continue
                
            # Calculate similarity (Hamming distance) between child and each previous step language
            def hamming_distance(id1, id2):
                """Calculate Hamming distance between two language IDs (treated as binary)"""
                xor = id1 ^ id2
                return bin(xor).count('1')
            
            # Find the most similar language (minimum Hamming distance)
            best_parent_id = min(prev_step_languages, key=lambda lang: hamming_distance(child_id, lang))
            
            # Update the parent_id for this event
            parent_id = best_parent_id
            current_parent_node = language_lineage_map.get(parent_id)
            
            if current_parent_node is None:
                continue
        
        if current_parent_node not in all_nodes:
            continue
        
        # Create internal node for the branching event
        internal_id = internal_node_id
        internal_node_id += 1
        
        # The internal node inherits the parent of the current parent node
        parent_of_internal = all_nodes[current_parent_node]['parent']
        
        all_nodes[internal_id] = {
            'birth_step': branch_time,
            'death_step': branch_time,  # Internal nodes exist at a single point
            'parent': parent_of_internal,
            'children': [],
            'is_leaf': False
        }
        
        # Create continuation of parent lineage
        parent_continuation_id = internal_node_id
        internal_node_id += 1
        
        # Get parent death time from original data
        if parent_id in [event['parent_id'] for event in branching_events]:
            parent_death_step = next(e['parent_death'] for e in branching_events if e['parent_id'] == parent_id)
        else:
            # Use the death time from language_info
            parent_data = language_info[language_info['language_id'] == parent_id].iloc[0]
            parent_death_step = parent_data['death_step']
        
        all_nodes[parent_continuation_id] = {
            'birth_step': branch_time,
            'death_step': parent_death_step,
            'parent': internal_id,
            'children': [],
            'is_leaf': True,
            'continuation': True,
            'original_id': parent_id
        }
        
        # Add child
        child_data = language_info[language_info['language_id'] == child_id].iloc[0]
        all_nodes[child_id] = {
            'birth_step': branch_time,
            'death_step': child_data['death_step'],
            'parent': internal_id,
            'children': [],
            'is_leaf': True
        }
        
        # Update parent relationships in the tree
        if parent_of_internal == 'ROOT':
            # This internal node becomes a direct child of ROOT, replacing the original parent
            pass  # ROOT's children will be computed later
        elif parent_of_internal in all_nodes:
            # Remove the old parent node from its parent's children and add the internal node
            if current_parent_node in all_nodes[parent_of_internal]['children']:
                all_nodes[parent_of_internal]['children'].remove(current_parent_node)
            all_nodes[parent_of_internal]['children'].append(internal_id)
        
        # Update the current parent node to be non-leaf and set its death time
        all_nodes[current_parent_node]['death_step'] = branch_time
        all_nodes[current_parent_node]['is_leaf'] = False
        # IMPORTANT: Update the parent of the original language to be the internal node
        all_nodes[current_parent_node]['parent'] = internal_id
        
        # Add children to internal node
        all_nodes[internal_id]['children'] = [parent_continuation_id, child_id]
        
        # Update lineage mapping
        language_lineage_map[parent_id] = parent_continuation_id
        language_lineage_map[child_id] = child_id
    
    # Build root's children list by finding all nodes with parent='ROOT'
    root_children_nodes = [node_id for node_id, node in all_nodes.items() if node['parent'] == 'ROOT']

    # Detailed breakdown
    initial_still_at_root = [node_id for node_id in root_children_nodes if node_id in initial_languages]
    internal_at_root = [node_id for node_id in root_children_nodes if node_id not in initial_languages]
    
    print(f"Initial languages still at ROOT: {len(initial_still_at_root)} - {initial_still_at_root[:5]}")
    print(f"Internal nodes at ROOT: {len(internal_at_root)} - {internal_at_root[:5]}")
    
    # Check for orphaned initial languages that should have been replaced
    orphaned_at_root = []
    for lang_id in initial_languages:
        if all_nodes[lang_id]['parent'] == 'ROOT' and all_nodes[lang_id]['is_leaf'] == False:
            orphaned_at_root.append(lang_id)
    
    if orphaned_at_root:
        print(f"WARNING: Found {len(orphaned_at_root)} initial languages that became parents but are still at ROOT!")
        print(f"These should have been replaced by internal nodes: {orphaned_at_root}")
    
    def build_newick_from_nodes(node_id, all_nodes, parent_time=None):
        """Build Newick string from node structure"""
        if node_id not in all_nodes:
            return str(node_id)
        
        node = all_nodes[node_id]
        children = node['children']
        
        if not children:
            # Leaf node
            if 'original_id' in node:
                # This is a continuation - make it unique by adding suffix
                label = f"{node['original_id']}_cont_{node_id}"
            else:
                label = str(node_id)
            
            # Branch length is from parent time to death time
            if parent_time is not None:
                branch_length = node['death_step'] - parent_time
            else:
                branch_length = node['death_step'] - node['birth_step']
            
            return f"{label}:{branch_length}"
        
        # Internal node with children
        child_strings = []
        for child_id in children:
            child_newick = build_newick_from_nodes(child_id, all_nodes, node['birth_step'])
            child_strings.append(child_newick)
        
        # For internal nodes, add the internal node label
        internal_label = f"INT_{node_id}"
        
        # For internal nodes, if we have a parent_time, add branch length
        if parent_time is not None:
            branch_length = node['birth_step'] - parent_time
            return f"({','.join(child_strings)}){internal_label}:{branch_length}"
        else:
            # Root level internal node
            return f"({','.join(child_strings)}){internal_label}"
    
    # Build the complete tree
    root_children = []
    
    # Add initial languages and root-level internal nodes
    for node_id, node in all_nodes.items():
        if node['parent'] == 'ROOT':
            if node_id in initial_languages:
                # Initial languages have branch length from first_step to death
                child_newick = build_newick_from_nodes(node_id, all_nodes, first_step)
            else:
                # Internal nodes at root level
                child_newick = build_newick_from_nodes(node_id, all_nodes, first_step)
            root_children.append(child_newick)
    
    if root_children:
        tree_string = f"({','.join(root_children)})ROOT;"
    else:
        # Fallback: create simple star tree
        all_languages = sorted([str(lang) for lang in df['language_id'].unique()])
        tree_string = f"({','.join(all_languages)})ROOT;"
    
    # Count relationships for reporting
    parent_children = defaultdict(list)
    for event in branching_events:
        parent_children[event['parent_id']].append(event['child_id'])
    
    # Return both tree string and internal node mapping for clade generation
    internal_node_mapping = {}
    for node_id, node in all_nodes.items():
        if not node['is_leaf']:  # Internal nodes
            internal_node_mapping[node_id] = f"INT_{node_id}"
    
    return tree_string, parent_children, internal_node_mapping

def export_to_newick(df, output_path):
    """Export phylogeny data to Newick format"""
    
    # Build hierarchical tree
    tree_string, parent_children, internal_node_mapping = build_hierarchical_tree(df)
    
    # Save to file
    newick_dir = os.path.join(os.path.dirname(output_path), "newick")
    os.makedirs(newick_dir, exist_ok=True)
    newick_file_path = os.path.join(newick_dir, os.path.basename(output_path).replace('.nex', '.nwk'))
    
    with open(newick_file_path, 'w') as f:
        f.write(tree_string)
    
    print(f"Newick file saved to {newick_file_path}")
    print(f"Tree structure created with {len(parent_children)} parent-child relationships")
    
    return newick_file_path

def export_binary_dataset(df, output_path, B=16):
    """Export binary dataset for iTOL visualization of language bitstrings"""
    
    # First, we need to get the tree string to extract all node IDs
    tree_string, _, _ = build_hierarchical_tree(df)
    
    # Pattern to match node labels (everything before a colon, excluding parentheses and commas)
    node_pattern = r'([^(),:\s]+):'
    all_node_ids = re.findall(node_pattern, tree_string)
    
    # Also extract ROOT and any nodes without branch lengths
    root_pattern = r'\)([^(),:\s]+)[;)]'
    root_nodes = re.findall(root_pattern, tree_string)
    all_node_ids.extend(root_nodes)
    
    print(f"Found {len(all_node_ids)} nodes in tree: {sorted(set(all_node_ids))[:10]}...")
    
    # Create mapping from tree node IDs to original language IDs for bitstring lookup
    tree_to_language_mapping = {}
    
    for node_id in set(all_node_ids):
        if node_id == 'ROOT':
            continue
            
        # Handle continuation nodes (extract original language ID)
        if '_cont_' in str(node_id):
            original_lang_id = str(node_id).split('_cont_')[0]
            if original_lang_id.isdigit():
                tree_to_language_mapping[node_id] = int(original_lang_id)
        else:
            # Regular node - use as is if it's a valid language ID
            if str(node_id).isdigit():
                tree_to_language_mapping[node_id] = int(node_id)
    
    print(f"Created mapping for {len(tree_to_language_mapping)} tree nodes to language IDs")
    
    # Create binary dataset content
    binary_content = """DATASET_BINARY
#Binary datasets are visualized as filled or empty symbols, depending on the value associated with a node (0 or 1).
#Each node can have multiple associated values, and each value will be represented by a symbol (defined in FIELD_SHAPES) with corresponding color and label (from FIELD_COLORS and FIELD_LABELS).
#Possible values (defined under DATA below) for each node are 1 (filled shapes), 0 (empty shapes) and -1 (completely omitted).

#=================================================================#
#                    MANDATORY SETTINGS                           #
#=================================================================#
SEPARATOR COMMA

#label is used in the legend table (can be changed later)
DATASET_LABEL,Language Bitstrings

#dataset color (can be changed later)
COLOR,#2166ac

#shapes for each field column; possible choices are
#1: rectangle 
#2: circle
#3: star
#4: right pointing triangle
#5: left pointing triangle
#6: check mark
"""
    
    # Generate field shapes (all rectangles)
    field_shapes = ",".join(["1"] * B)
    binary_content += f"FIELD_SHAPES,{field_shapes}\n\n"
    
    # Generate field labels (bit positions)
    field_labels = ",".join([f"bit_{i}" for i in range(B)])
    binary_content += f"FIELD_LABELS,{field_labels}\n\n"
    
    # Add optional settings
    binary_content += """#=================================================================#
#                    OPTIONAL SETTINGS                            #
#=================================================================#

#define colors for each individual field column
"""
    
    # Generate alternating colors for better visibility
    colors = []
    for i in range(B):
        if i % 2 == 0:
            colors.append("#2166ac")  # Blue
        else:
            colors.append("#762a83")  # Purple
    
    field_colors = ",".join(colors)
    binary_content += f"FIELD_COLORS,{field_colors}\n\n"
    
    binary_content += """#show dashed lines between leaf labels and the dataset
DASHED_LINES,1

#left margin, used to increase/decrease the spacing to the next dataset
MARGIN,0

#symbol height factor
HEIGHT_FACTOR,1.2

#increase/decrease the spacing between individual levels
SYMBOL_SPACING,-15

#display the text labels above each field column
SHOW_LABELS,0

#text label size factor
SIZE_FACTOR,0.8

#text label rotation
LABEL_ROTATION,45

HORIZONTAL_GRID,0
VERTICAL_GRID,0

#=================================================================#
#       Actual data follows after the "DATA" keyword              #
#=================================================================#
DATA
"""
    
    # Generate data for each tree node that maps to a language
    for tree_node_id, lang_id in sorted(tree_to_language_mapping.items()):
        # Convert language_id (integer) to binary representation
        binary_str = format(int(lang_id), f'0{B}b')
        
        # Create comma-separated values for each bit
        bit_values = ",".join(binary_str)
        
        # Use the exact tree node ID
        binary_content += f"{tree_node_id},{bit_values}\n"
    
    # Save to file
    binary_dir = os.path.join(os.path.dirname(output_path), "itol_datasets")
    os.makedirs(binary_dir, exist_ok=True)
    binary_file_path = os.path.join(binary_dir, os.path.basename(output_path).replace('.nwk', '_binary.txt'))
    
    with open(binary_file_path, 'w') as f:
        f.write(binary_content)
    
    print(f"Binary dataset file saved to {binary_file_path}")
    print(f"Dataset contains bitstring visualization for {len(tree_to_language_mapping)} tree nodes")
    
    return binary_file_path

def export_clade_dataset(df, output_path):
    """Export clade dataset for iTOL visualization of initial language clades"""
    import numpy as np
    
    # Build the tree structure to get node relationships
    tree_string, _, internal_node_mapping = build_hierarchical_tree(df)
    
    # We need to rebuild the actual tree structure from build_hierarchical_tree
    # to get the correct parent-child relationships
    
    # Get language information
    language_births = df.groupby('language_id').agg({
        'step': 'min',
        'parent_id': 'first'
    }).reset_index()
    language_births.rename(columns={'step': 'birth_step'}, inplace=True)
    
    language_deaths = df.groupby('language_id')['step'].max().reset_index()
    language_deaths.rename(columns={'step': 'death_step'}, inplace=True)
    
    language_info = language_births.merge(language_deaths, on='language_id')
    language_info['parent_id_clean'] = pd.to_numeric(language_info['parent_id'], errors='coerce')
    
    first_step = df['step'].min()
    initial_languages = df[df['step'] == first_step]['language_id'].unique()
    
    # Rebuild the EXACT same tree structure as in build_hierarchical_tree
    internal_node_id = max(df['language_id'].max(), 1000) + 1
    all_nodes = {}
    language_lineage_map = {}
    
    # Add initial languages as children of root
    for lang_id in initial_languages:
        lang_data = language_info[language_info['language_id'] == lang_id].iloc[0]
        all_nodes[lang_id] = {
            'birth_step': lang_data['birth_step'],
            'death_step': lang_data['death_step'],
            'parent': 'ROOT',
            'children': [],
            'is_leaf': True
        }
        language_lineage_map[lang_id] = lang_id
    
    # Process branching events exactly as in build_hierarchical_tree
    branching_events = []
    for _, lang_data in language_info.iterrows():
        lang_id = lang_data['language_id']
        parent_id = lang_data['parent_id_clean']
        birth_step = lang_data['birth_step']
        
        if lang_id in initial_languages or pd.isna(parent_id):
            continue
        if parent_id not in language_info['language_id'].values:
            continue
            
        parent_data = language_info[language_info['language_id'] == parent_id].iloc[0]
        branching_events.append({
            'parent_id': int(parent_id),
            'child_id': int(lang_id),
            'branch_time': birth_step,
            'parent_birth': parent_data['birth_step'],
            'parent_death': parent_data['death_step'],
            'child_birth': birth_step,
            'child_death': lang_data['death_step']
        })
    
    branching_events.sort(key=lambda x: x['branch_time'])
    
    # Process each branching event exactly as in build_hierarchical_tree
    for event in branching_events:
        parent_id = event['parent_id']
        child_id = event['child_id']
        branch_time = event['branch_time']
        
        current_parent_node = language_lineage_map.get(parent_id)
        
        if current_parent_node is None:
            # Find most similar language from previous step
            prev_step = branch_time - 500
            prev_step_languages = df[df['step'] == prev_step]['language_id'].unique()
            
            if len(prev_step_languages) == 0:
                continue
                
            def hamming_distance(id1, id2):
                """Calculate Hamming distance between two language IDs (treated as binary)"""
                xor = id1 ^ id2
                return bin(xor).count('1')
            
            best_parent_id = min(prev_step_languages, key=lambda lang: hamming_distance(child_id, lang))
            parent_id = best_parent_id
            current_parent_node = language_lineage_map.get(parent_id)
            
            if current_parent_node is None:
                continue
        
        if current_parent_node not in all_nodes:
            continue
        
        # Create internal node
        internal_id = internal_node_id
        internal_node_id += 1
        
        parent_of_internal = all_nodes[current_parent_node]['parent']
        
        all_nodes[internal_id] = {
            'birth_step': branch_time,
            'death_step': branch_time,  # Internal nodes exist at a single point
            'parent': parent_of_internal,
            'children': [],
            'is_leaf': False
        }
        
        # Create continuation of parent lineage
        parent_continuation_id = internal_node_id
        internal_node_id += 1
        
        # Get parent death time from original data
        if parent_id in [event['parent_id'] for event in branching_events]:
            parent_death_step = next(e['parent_death'] for e in branching_events if e['parent_id'] == parent_id)
        else:
            # Use the death time from language_info
            parent_data = language_info[language_info['language_id'] == parent_id].iloc[0]
            parent_death_step = parent_data['death_step']
        
        all_nodes[parent_continuation_id] = {
            'birth_step': branch_time,
            'death_step': parent_death_step,
            'parent': internal_id,
            'children': [],
            'is_leaf': True,
            'continuation': True,
            'original_id': parent_id
        }
        
        # Add child
        child_data = language_info[language_info['language_id'] == child_id].iloc[0]
        all_nodes[child_id] = {
            'birth_step': branch_time,
            'death_step': child_data['death_step'],
            'parent': internal_id,
            'children': [],
            'is_leaf': True
        }
        
        # Update parent relationships in the tree
        if parent_of_internal == 'ROOT':
            # This internal node becomes a direct child of ROOT, replacing the original parent
            pass  # ROOT's children will be computed later
        elif parent_of_internal in all_nodes:
            # Remove the old parent node from its parent's children and add the internal node
            if current_parent_node in all_nodes[parent_of_internal]['children']:
                all_nodes[parent_of_internal]['children'].remove(current_parent_node)
            all_nodes[parent_of_internal]['children'].append(internal_id)
        
        # Update the current parent node to be non-leaf and set its death time
        all_nodes[current_parent_node]['death_step'] = branch_time
        all_nodes[current_parent_node]['is_leaf'] = False
        # IMPORTANT: Update the parent of the original language to be the internal node
        all_nodes[current_parent_node]['parent'] = internal_id
        
        # Add children to internal node
        all_nodes[internal_id]['children'] = [parent_continuation_id, child_id]
        
        # Update lineage mapping
        language_lineage_map[parent_id] = parent_continuation_id
        language_lineage_map[child_id] = child_id
    
    # Find ROOT's direct children (these define our clades)
    root_children = [node_id for node_id, node in all_nodes.items() if node['parent'] == 'ROOT']
    num_clades = len(root_children)
    
    print(f"Found {num_clades} clades (direct children of ROOT): {root_children}")
    
    # Map root children to their internal node labels (if they are internal nodes)
    root_children_labels = []
    for node_id in root_children:
        if node_id in internal_node_mapping:
            # This is an internal node, use its label
            root_children_labels.append(internal_node_mapping[node_id])
        else:
            # This is a leaf node, use the node ID directly
            root_children_labels.append(str(node_id))
    
    print(f"Root children labels for clades: {root_children_labels}")
    
    # Generate rainbow colors
    def generate_rainbow_colors(n):
        colors = []
        for i in range(n):
            hue = i / n
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            colors.append(hex_color)
        return colors

    rainbow_colors = generate_rainbow_colors(num_clades)
    np.random.shuffle(rainbow_colors)

    # Create clade dataset content using internal node labels
    clade_content = """DATASET_RANGE
#Colored/labeled range datasets allow the highlighting of various clades or leaf ranges by using colored boxes or brackets.

#=================================================================#
#                    MANDATORY SETTINGS                           #
#=================================================================#
SEPARATOR COMMA

#label is used in the legend table (can be changed later)
DATASET_LABEL,Language Clades

#dataset color in the legend table
COLOR,#ff0000

#=================================================================#
#                    OPTIONAL SETTINGS                            #
#=================================================================#

#RANGE_TYPE defines how the ranges will be visualized: 'box' or 'bracket'
RANGE_TYPE,box

#specify what the range boxes will cover: 'label','clade' or 'tree'
RANGE_COVER,tree

#simplify or smooth polygons when in unrooted display mode
UNROOTED_SMOOTH,simplify

SHOW_LABELS,0

#when RANGE_COVER is set to 'clade' or 'tree', you can disable the covering of labels
COVER_LABELS,1

#if set to 1, ranges will cover any displayed external datasets as well
COVER_DATASETS,0

#Options related to range labels
SHOW_LABELS,0


#=================================================================#
#       Actual data follows after the "DATA" keyword              #
#=================================================================#
#START_NODE_ID,END_NODE_ID,FILL_COLOR,GRADIENT_COLOR,LINE_COLOR,LINE_STYLE,LINE_WIDTH,LABEL_TEXT,LABEL_COLOR,LABEL_SIZE_FACTOR,LABEL_STYLE

DATA
"""
    
    # Add range definitions for each clade using the internal node labels
    for i, clade_label in enumerate(root_children_labels):
        color = rainbow_colors[i % len(rainbow_colors)]
        
        # Add alpha to color (33 in hex ≈ 20% alpha)
        fill_color = f"{color}33"
        
        # Use the internal node label as both START and END to highlight the entire clade
        clade_content += f"{clade_label},{clade_label},{fill_color},,{color},solid,1,Clade {i+1},{color},1,normal\n"
    
    # Save to file
    clade_dir = os.path.join(os.path.dirname(output_path), "itol_datasets")
    os.makedirs(clade_dir, exist_ok=True)
    clade_file_path = os.path.join(clade_dir, os.path.basename(output_path).replace('.nwk', '_clades.txt'))
    
    with open(clade_file_path, 'w') as f:
        f.write(clade_content)
    
    print(f"Clade dataset file saved to {clade_file_path}")
    print(f"Dataset contains {num_clades} clades with rainbow coloring using single node method")
    
    # Print detailed clade summary
    def get_all_descendants(node_id, all_nodes):
        """Get all descendant nodes of a given node"""
        descendants = []
        if node_id in all_nodes:
            descendants.append(node_id)
            for child in all_nodes[node_id]['children']:
                descendants.extend(get_all_descendants(child, all_nodes))
        return descendants
    
    for i, (clade_root, clade_label) in enumerate(zip(root_children, root_children_labels)):
        descendants = get_all_descendants(clade_root, all_nodes)
        leaf_descendants = [node for node in descendants if node in all_nodes and all_nodes[node]['is_leaf']]
        print(f"Clade {i+1} (root: {clade_root}, label: {clade_label}): {len(descendants)} total nodes, {len(leaf_descendants)} leaf nodes, color: {rainbow_colors[i % len(rainbow_colors)]}")
        print(f"  Sample leaf nodes: {sorted([str(n) for n in leaf_descendants], key=lambda x: str(x))[:10]}{'...' if len(leaf_descendants) > 10 else ''}")
    
    return clade_file_path

def convert_to_newick(tsv_path, output_path=None):
    """Convert TSV phylogeny data to Newick format with binary dataset"""
    # Load data
    df = load_phylogeny_data(tsv_path)
    print(f"Loaded {len(df)} records from {tsv_path}")
    
    # Analyze population dynamics
    analyze_population_dynamics(df)
    
    # Create population plot
    output_dir = os.path.dirname(output_path) if output_path else os.path.dirname(tsv_path)
    plot_population_over_time(df, output_dir)
    
    print(f"\nBuilding phylogenetic tree...")
    
    # Export to Newick format
    if output_path:
        newick_path = export_to_newick(df, output_path)
        
        # Export binary dataset for iTOL
        print(f"\nGenerating binary dataset for iTOL...")
        binary_path = export_binary_dataset(df, output_path)
        
        # Export clade dataset for iTOL
        print(f"\nGenerating clade dataset for iTOL...")
        clade_path = export_clade_dataset(df, output_path)
        
        return newick_path, binary_path, clade_path

def main():
    # Define paths
    tsv_file = "/home/rizfn/github/babel-transition/src/understandabilityVsHamming2D/stochasticCommutable/outputs/phylogeny/L_256_g_1_a_1_B_16_mu_0.001_minSize_10.tsv"
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(tsv_file))[0]
    output_dir = os.path.dirname(tsv_file)
    newick_file = os.path.join(output_dir, f"{base_name}.nwk")
    
    try:
        # Convert to Newick format and analyze
        convert_to_newick(tsv_file, newick_file)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
