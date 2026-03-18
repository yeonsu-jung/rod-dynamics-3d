import numpy as np
import argparse

def generate_weaving(filename, d, box_size, ar=None):
    """
    Generates a weaving structure with:
    - Layer 0 (z=0, 2d..): X-aligned rods, spacing d (pitch 2d)
    - Layer 1 (z=d, 3d..): Y-aligned rods, spacing d (pitch 2d)
    - Vertical rods (z-axis): Filling gaps at (x,y) = (odd*d, odd*d)
    """
    
    # Determine rod length
    if ar is None:
        rod_length = box_size
    else:
        rod_length = ar * d
        
    print(f"Generating weaving structure with d={d}, box_size={box_size}, rod_length={rod_length}")
    
    rods = []
    
    # Box bounds (centered at 0)
    half_box = box_size / 2.0
    start = -half_box
    end = half_box
    
    # Rod extents relative to center 0
    half_L = rod_length / 2.0
    
    # Pitch for horizontal layers
    # Rod thickness d, gap d -> Pitch = 2d
    pitch = 2 * d
    
    # Coordinate generators
    # We want to cover the box.
    # Grid lines: k * pitch. 
    # We centre the grid at 0.
    
    # Calculate range of indices to cover the box
    # min_idx * pitch >= start  --> min_idx >= start/pitch
    min_idx = int(np.floor(start / pitch)) - 1
    max_idx = int(np.ceil(end / pitch)) + 1
    
    indices = range(min_idx, max_idx + 1)
    
    # Horizontal grid positions (rod centers)
    grid_coords = [i * pitch for i in indices if start <= i * pitch <= end]
    
    # Gap positions (vertical rod centers)
    # Gap is at grid_coord + d 
    gaps = [(i * pitch) + d for i in indices]
    gap_coords = [v for v in gaps if start <= v <= end]
    
    # ---------------------------------------------------------
    # Generate Horizontal Layers (X and Y aligned)
    # ---------------------------------------------------------
    
    # Z-levels: k * d
    # If box extends in Z, fill it.
    z_min_idx = int(np.floor(start / d))
    z_max_idx = int(np.ceil(end / d))
    
    for k in range(z_min_idx, z_max_idx + 1):
        z_level = k * d
        if z_level < start or z_level > end:
            continue
            
        # Event k -> Layer 0 (X-aligned) or similar
        # Odd k  -> Layer 1 (Y-aligned)
        
        if k % 2 == 0:
            # X-aligned rods
            # Spaced along Y at grid_coords
            for y in grid_coords:
                # Rod endpoints
                # Centered at 0, spanning [-half_L, half_L]
                p0 = [-half_L, y, z_level]
                p1 = [half_L, y, z_level]
                rods.append(p0 + p1) # flatten
        else:
            # Y-aligned rods
            # Spaced along X at grid_coords
            for x in grid_coords:
                # Rod endpoints
                p0 = [x, -half_L, z_level]
                p1 = [x, half_L, z_level]
                rods.append(p0 + p1)

    # ---------------------------------------------------------
    # Generate Vertical Rods (Z-aligned)
    # ---------------------------------------------------------
    
    # Located at intersection of gaps
    for x in gap_coords:
        for y in gap_coords:
            p0 = [x, y, -half_L]
            p1 = [x, y, half_L]
            rods.append(p0 + p1)

    print(f"Total rods generated: {len(rods)}")
    
    # Write to CSV
    with open(filename, 'w') as f:
        # Header comments
        f.write(f"# rod_diameter={d}\n")
        f.write(f"# box_size={box_size}\n")
        f.write(f"# rod_length={rod_length}\n")
        f.write(f"# placed={len(rods)}\n")
        f.write(f"# aspect_ratio={rod_length/d}\n")
        f.write("x0,y0,z0,x1,y1,z1\n")
        
        for r in rods:
            f.write(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weaving rod configuration")
    parser.add_argument("filename", default="weaving.csv", nargs="?")
    parser.add_argument("--d", type=float, default=0.05, help="Rod thickness and gap size")
    parser.add_argument("--box", type=float, default=0.5, help="Box size")
    parser.add_argument("--ar", type=float, default=None, help="Rod aspect ratio (Length/Diameter). If not set, Length=BoxSize")
    
    args = parser.parse_args()
    generate_weaving(args.filename, args.d, args.box, args.ar)
