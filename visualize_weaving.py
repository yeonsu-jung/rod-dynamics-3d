import polyscope as ps
import numpy as np
import argparse
import csv

def visualize_weaving(filename):
    # Read CSV
    rods = []
    diameter = 0.01 # Default
    
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('#'):
            if 'rod_diameter' in line:
                try:
                    diameter = float(line.split('=')[1].strip())
                except:
                    pass
            line = f.readline()
        
        # Now we are at the header or data
        # If the last read line was the header "x0,y0...", we continue
        # If it was data, we process it (but usually the loop above consumes comments)
        
        reader = csv.reader(f)
        # The while loop above might have consumed the header if it didn't start with #
        # standard format has header "x0,y0,z0,x1,y1,z1"
        
        # if the line variable is not the header, and not a comment, acts as first row?
        # Re-opening to be safe and using csv module properly skipping comments
        pass

    # Re-reading cleanly
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'rod_diameter' in line:
                     diameter = float(line.split('=')[1].strip())
                continue
            if line.startswith('x0'):
                continue
            
            parts = [float(x) for x in line.split(',')]
            data.append(parts)
    
    data = np.array(data) # shape (N, 6)
    
    if len(data) == 0:
        print("No rods found in file.")
        return

    print(f"Loaded {len(data)} rods. Diameter: {diameter}")

    # Prepare nodes and edges for Curve Network
    # Nodes: (2N, 3)
    # Edges: (N, 2)
    
    p0 = data[:, 0:3]
    p1 = data[:, 3:6]
    
    nodes = np.zeros((2 * len(data), 3))
    nodes[0::2] = p0
    nodes[1::2] = p1
    
    edges = np.column_stack((np.arange(0, 2*len(data), 2), np.arange(1, 2*len(data), 2)))

    # Initialize Polyscope
    ps.init()
    
    ps.set_up_dir("z_up")
    
    # Register curve network
    ps_net = ps.register_curve_network("Weaving Structure", nodes, edges)
    ps_net.set_radius(diameter/2.0, relative=False)
    
    ps.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize weaving structure with Polyscope")
    parser.add_argument("filename", default="weaving.csv", nargs="?")
    args = parser.parse_args()
    
    visualize_weaving(args.filename)
