import polyscope as ps
import numpy as np
import argparse
import os

def visualize_entangled(filename, diameter, screenshot_path=None):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    # ... (data loading logic same as before)
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x) for x in line.split()]
            if len(parts) == 6:
                data.append(parts)
    
    data = np.array(data)
    if len(data) == 0:
        return

    # Endpoint columns: x0 y0 z0 x1 y1 z1
    p0 = data[:, 0:3]
    p1 = data[:, 3:6]
    
    nodes = np.zeros((2 * len(data), 3))
    nodes[0::2] = p0
    nodes[1::2] = p1
    edges = np.column_stack((np.arange(0, 2*len(data), 2), np.arange(1, 2*len(data), 2)))

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_background_color((1.0, 1.0, 1.0)) # Default white for comparison
    
    ps_net = ps.register_curve_network("Entangled Packing", nodes, edges)
    ps_net.set_radius(diameter/2.0, relative=False)
    
    # Assign per-rod colors (categorical)
    # Different colors, maybe 10
    colors = np.zeros((len(data), 3))
    palette = np.array([
        [0.30, 0.70, 1.00], # Blue
        [1.00, 0.55, 0.25], # Orange
        [0.60, 0.90, 0.40], # Green
        [0.90, 0.40, 0.80], # Purple
        [0.95, 0.85, 0.30], # Yellow
        [0.90, 0.30, 0.30], # Red
        [0.30, 0.90, 0.90], # Cyan
        [1.00, 0.60, 0.80], # Pink
        [0.60, 0.40, 0.20], # Brown
        [0.50, 0.50, 0.50]  # Gray
    ])
    for i in range(len(data)):
        colors[i] = palette[i % len(palette)]
    
    # We need to map per-rod colors to edges in Polyscope
    # Each edge represents one rod
    ps_net.add_color_quantity("Rod Colors", colors, defined_on='edges', enabled=True)
    
    if screenshot_path:
        ps.set_view_projection_mode("perspective")
        # Match the -1, 1, -1 angle roughly or use ps.look_at
        # Simplified: just auto-center and screenshot
        ps.screenshot(screenshot_path)
    else:
        ps.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize entangled packings with Polyscope")
    parser.add_argument("filename", help="Path to x_entangled.txt")
    parser.add_argument("--diameter", type=float, default=0.01, help="Diameter of the rods")
    parser.add_argument("--screenshot", help="Path to save screenshot and exit")
    args = parser.parse_args()
    
    visualize_entangled(args.filename, args.diameter, args.screenshot)
