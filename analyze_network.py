import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import sys
import os

def analyze_network(csv_path, perrod_path=None, output_prefix="network_analysis"):
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading network CSV: {e}")
        return

    # 1. Network / Cluster Analysis
    print("Performing cluster analysis...")
    frames = sorted(df['frame'].unique())
    cluster_results = []
    
    # We can do cluster analysis frame-by-frame
    # Optimize by grouping
    grouped = df.groupby('frame')
    for frame, group in grouped:
        G = nx.Graph()
        edges = list(zip(group['rod_i'], group['rod_j']))
        if edges:
            G.add_edges_from(edges)
            components = list(nx.connected_components(G))
            num_clusters = len(components)
            max_cluster_size = max(len(c) for c in components)
            num_active_nodes = G.number_of_nodes()
        else:
            num_clusters = 0
            max_cluster_size = 0
            num_active_nodes = 0
            
        cluster_results.append({
            'frame': frame,
            'num_clusters': num_clusters,
            'max_cluster_size': max_cluster_size,
            'num_active_nodes': num_active_nodes
        })
    
    df_clusters = pd.DataFrame(cluster_results)
    stats_csv = f"{output_prefix}_stats.csv"
    df_clusters.to_csv(stats_csv, index=False)
    print(f"Cluster statistics saved to {stats_csv}")
    
    # Plotting Clusters
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(df_clusters['frame'], df_clusters['num_clusters'])
    plt.title('Num Clusters')
    plt.subplot(1, 3, 2)
    plt.plot(df_clusters['frame'], df_clusters['max_cluster_size'], color='orange')
    plt.title('Max Cluster Size')
    plt.subplot(1, 3, 3)
    plt.plot(df_clusters['frame'], df_clusters['num_active_nodes'], color='green')
    plt.title('Active Rods')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png")
    print(f"Plot saved to {output_prefix}.png")

    # 2. Net Force and Torque Calculation
    print("\nCalculating Net Forces...")
    
    # Prepare side A
    cols_a = {
        'frame': 'frame', 'rod_i': 'rod', 
        'force_a_x': 'fx', 'force_a_y': 'fy', 'force_a_z': 'fz',
        'friction_a_x': 'frx', 'friction_a_y': 'fry', 'friction_a_z': 'frz',
        'contact_x': 'cx', 'contact_y': 'cy', 'contact_z': 'cz'
    }
    df_a = df[list(cols_a.keys())].rename(columns=cols_a)
    
    # Prepare side B
    cols_b = {
        'frame': 'frame', 'rod_j': 'rod', 
        'force_b_x': 'fx', 'force_b_y': 'fy', 'force_b_z': 'fz',
        'friction_b_x': 'frx', 'friction_b.y': 'fry', 'friction_b.z': 'frz', # Note typical column naming issue .y vs _y
        'contact_x': 'cx', 'contact_y': 'cy', 'contact_z': 'cz'
    }
    # Check for .y vs _y in input
    if 'friction_b.y' not in df.columns and 'friction_b_y' in df.columns:
        cols_b['friction_b_y'] = 'fry'
        del cols_b['friction_b.y']
    if 'friction_b.z' not in df.columns and 'friction_b_z' in df.columns:
        cols_b['friction_b_z'] = 'frz'
        del cols_b['friction_b.z']
        
    df_b = df[list(cols_b.keys())].rename(columns=cols_b)
    
    # Combine
    df_forces = pd.concat([df_a, df_b], ignore_index=True)
    
    # Total force per contact point per rod = Normal Force + Friction Force
    df_forces['total_fx'] = df_forces['fx'] + df_forces['frx']
    df_forces['total_fy'] = df_forces['fy'] + df_forces['fry']
    df_forces['total_fz'] = df_forces['fz'] + df_forces['frz']
    
    # If perrod data is available, compute Torque
    if perrod_path and os.path.exists(perrod_path):
        print(f"Loading rod positions from {perrod_path}...")
        try:
            df_rods = pd.read_csv(perrod_path)
            # Ensure columns exist
            required_rod_cols = {'frame', 'rod', 'px', 'py', 'pz'}
            if required_rod_cols.issubset(df_rods.columns):
                # Merge logic
                # Optimization: Filter df_rods to frames present in df_forces
                target_frames = df_forces['frame'].unique()
                df_rods = df_rods[df_rods['frame'].isin(target_frames)]
                
                merged = pd.merge(df_forces, df_rods[['frame', 'rod', 'px', 'py', 'pz']], 
                                  on=['frame', 'rod'], how='left')
                
                # Calculate Lever Arm (r = contact - COM)
                merged['rx'] = merged['cx'] - merged['px']
                merged['ry'] = merged['cy'] - merged['py']
                merged['rz'] = merged['cz'] - merged['pz']
                
                # Torque = r x F
                merged['tx'] = merged['ry'] * merged['total_fz'] - merged['rz'] * merged['total_fy']
                merged['ty'] = merged['rz'] * merged['total_fx'] - merged['rx'] * merged['total_fz']
                merged['tz'] = merged['rx'] * merged['total_fy'] - merged['ry'] * merged['total_fx']
                
                df_forces = merged
            else:
                print(f"Warning: perrod CSV missing required columns {required_rod_cols}. Skipping torque.")
        except Exception as e:
            print(f"Error loading perrod CSV: {e}")
    else:
        print("No perrod file provided or found. Skipping torque calculation.")
        df_forces['tx'] = 0.0
        df_forces['ty'] = 0.0
        df_forces['tz'] = 0.0

    # Group by frame and rod to get Net Force/Torque
    net_df = df_forces.groupby(['frame', 'rod'])[['total_fx', 'total_fy', 'total_fz', 'tx', 'ty', 'tz']].sum().reset_index()
    net_df.columns = ['frame', 'rod', 'net_fx', 'net_fy', 'net_fz', 'net_tx', 'net_ty', 'net_tz']
    
    net_csv = f"{output_prefix}_forces.csv"
    net_df.to_csv(net_csv, index=False)
    print(f"Net forces and torques saved to {net_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze contact network and forces.")
    parser.add_argument("csv_file", help="Path to network.csv file")
    parser.add_argument("--per-rod", help="Path to perrod.csv file for torque calculation")
    parser.add_argument("--output", default="network_analysis", help="Output prefix")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    analyze_network(args.csv_file, args.per_rod, args.output)
