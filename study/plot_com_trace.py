import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def plot_com_trace(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    print(f"Loading CoM data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))

    # 1. 3D Trajectory (Top Left)
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(df['com_x'], df['com_y'], df['com_z'], label='CoM Trace', linewidth=1)
    ax1.scatter(df['com_x'].iloc[0], df['com_y'].iloc[0], df['com_z'].iloc[0], color='green', label='Start')
    ax1.scatter(df['com_x'].iloc[-1], df['com_y'].iloc[-1], df['com_z'].iloc[-1], color='red', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 2. Time Series (Top Center)
    ax2 = fig.add_subplot(232)
    ax2.plot(df['frame'], df['com_x'], label='X', alpha=0.7)
    ax2.plot(df['frame'], df['com_y'], label='Y', alpha=0.7)
    ax2.plot(df['frame'], df['com_z'], label='Z', alpha=0.7)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Position')
    ax2.set_title('Components over Time')
    ax2.legend()
    ax2.grid(True)

    # 3. XY Plane (Top Right)
    ax3 = fig.add_subplot(233)
    ax3.plot(df['com_x'], df['com_y'], linewidth=1)
    ax3.scatter(df['com_x'].iloc[0], df['com_y'].iloc[0], color='green', label='Start')
    ax3.scatter(df['com_x'].iloc[-1], df['com_y'].iloc[-1], color='red', label='End')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('XY Projection')
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='datalim')

    # 4. YZ Plane (Bottom Left)
    ax4 = fig.add_subplot(234)
    ax4.plot(df['com_y'], df['com_z'], linewidth=1)
    ax4.scatter(df['com_y'].iloc[0], df['com_z'].iloc[0], color='green', label='Start')
    ax4.scatter(df['com_y'].iloc[-1], df['com_z'].iloc[-1], color='red', label='End')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.grid(True)
    ax4.set_aspect('equal', adjustable='datalim')

    # 5. ZX Plane (Bottom Center)
    ax5 = fig.add_subplot(235)
    ax5.plot(df['com_z'], df['com_x'], linewidth=1)
    ax5.scatter(df['com_z'].iloc[0], df['com_x'].iloc[0], color='green', label='Start')
    ax5.scatter(df['com_z'].iloc[-1], df['com_x'].iloc[-1], color='red', label='End')
    ax5.set_xlabel('Z')
    ax5.set_ylabel('X')
    ax5.set_title('ZX Projection')
    ax5.grid(True)
    ax5.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    
    output_img = csv_path.replace('.csv', '.png')
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")
    # plt.show() # Uncomment if you want to see the plot interactively

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Center of Mass trajectory from CSV.")
    parser.add_argument("csv_file", nargs='?', default="com_debug.csv", help="Path to the CoM CSV file (default: com_debug.csv)")
    args = parser.parse_args()

    plot_com_trace(args.csv_file)
