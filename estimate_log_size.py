import argparse
import math

def estimate_log_size(N, steps, max_mb=100, Z=6):
    """
    Estimate log size and recommend stride.
    
    Args:
        N (int): Number of rods.
        steps (int): Total simulation steps.
        max_mb (float): Maximum allowed size in MB.
        Z (float): Estimated coordination number (avg contacts per rod).
    """
    
    # Per-rod data size estimation
    # Columns: frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,KE_rot,KE_total
    # 18 columns. Assuming ~15 bytes per float + commas + integers.
    # Let's be conservative: ~250 bytes per rod per frame.
    bytes_per_rod = 250
    frame_size_perrod = N * bytes_per_rod
    
    # Contact data size estimation
    # Columns: frame,stage,idx,a,b,px,py,pz,nx,ny,nz,pen,shiftBx,shiftBy,shiftBz,vn,vt
    # 17 columns. Assuming ~200 bytes per contact.
    # Number of contacts C = N * Z / 2
    C = N * Z / 2
    bytes_per_contact = 200
    frame_size_contact = C * bytes_per_contact
    
    total_bytes_per_frame = frame_size_perrod + frame_size_contact
    
    max_bytes = max_mb * 1024 * 1024
    
    max_frames = max_bytes / total_bytes_per_frame
    
    if max_frames < 1:
        stride = steps # Can't even log one frame?
    else:
        stride = math.ceil(steps / max_frames)
        
    # Ensure stride is at least 1
    stride = max(1, stride)
    
    estimated_size_mb = (steps / stride) * total_bytes_per_frame / (1024 * 1024)
    
    print(f"--- Estimation for N={N}, Steps={steps}, Max={max_mb}MB ---")
    print(f"Est. Per-Rod Frame Size: {frame_size_perrod / 1024:.2f} KB")
    print(f"Est. Contact Frame Size: {frame_size_contact / 1024:.2f} KB (assuming Z={Z})")
    print(f"Total Frame Size: {total_bytes_per_frame / 1024:.2f} KB")
    print(f"Max Frames allowed: {int(max_frames)}")
    print(f"Recommended Stride: {stride}")
    print(f"Est. Total Size: {estimated_size_mb:.2f} MB")
    
    return stride

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate log size and stride.")
    parser.add_argument("--N", type=int, required=True, help="Number of rods")
    parser.add_argument("--steps", type=int, required=True, help="Total simulation steps")
    parser.add_argument("--max_mb", type=float, default=100.0, help="Max size in MB")
    args = parser.parse_args()
    
    estimate_log_size(args.N, args.steps, args.max_mb)
