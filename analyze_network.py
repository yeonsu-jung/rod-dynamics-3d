import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import sys
import os

def analyze_network(csv_path, output_prefix="network_analysis"):
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    required_cols = {'frame', 'rod_i', 'rod_j'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV missing required columns: {required_cols - set(df.columns)}")
        return

    frames = sorted(df['frame'].unique())
    print(f"Found {len(frames)} frames. Processing...")

    results = []

    for frame in frames:
        df_frame = df[df['frame'] == frame]
        G = nx.Graph()
        # Add edges. Nodes are rod indices.
        edges = list(zip(df_frame['rod_i'], df_frame['rod_j']))
        if edges:
            G.add_edges_from(edges)
        
        # Connected components (clusters)
        if len(G) > 0:
            components = list(nx.connected_components(G))
            num_clusters = len(components)
            # Size of biggest cluster
            max_cluster_size = max(len(c) for c in components) if components else 0
            
            # Additional metric: Number of nodes involved in contact
            num_active_nodes = G.number_of_nodes()
        else:
            num_clusters = 0
            max_cluster_size = 0
            num_active_nodes = 0
        
        results.append({
            'frame': frame,
            'num_clusters': num_clusters,
            'max_cluster_size': max_cluster_size,
            'num_active_nodes': num_active_nodes
        })

    results_df = pd.DataFrame(results)
    
    # Save statistics
    stats_csv = f"{output_prefix}_stats.csv"
    results_df.to_csv(stats_csv, index=False)
    print(f"Statistics saved to {stats_csv}")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(results_df['frame'], results_df['num_clusters'], label='Num Clusters')
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('Number of Contact Clusters')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(results_df['frame'], results_df['max_cluster_size'], label='Max Size', color='orange')
    plt.xlabel('Frame')
    plt.ylabel('Size (Nodes)')
    plt.title('Size of Largest Cluster')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(results_df['frame'], results_df['num_active_nodes'], label='Active Nodes', color='green')
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('Total Rods in Contact')
    plt.grid(True)
    
    output_img = f"{output_prefix}.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")
    
    # Summary of last frame
    if not results_df.empty:
        last = results_df.iloc[-1]
        print("\nLast Frame Summary:")
        print(last)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze contact network evolution.")
    parser.add_argument("csv_file", help="Path to network.csv file")
    parser.add_argument("--output", default="network_analysis", help="Output prefix")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    analyze_network(args.csv_file, args.output)
