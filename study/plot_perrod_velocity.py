import csv
import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt

def load_perrod_csv(path: Path):
    frames = []
    velocities_sq = []
    rods = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            rod_id = int(row["rod"])
            vx = float(row["vx"])
            vy = float(row["vy"])
            vz = float(row["vz"])
            
            v_sq = vx*vx + vy*vy + vz*vz
            
            frames.append(frame)
            velocities_sq.append(v_sq)
            rods.append(rod_id)

    return {
        "frame": frames,
        "v_sq": velocities_sq,
        "rod": rods
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_perrod_velocity.py path/to/perrod.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    data = load_perrod_csv(csv_path)

    # Group by rod if multiple rods exist (though request implies single or we just plot all points)
    # The user asked for "velocity squared over time".
    # If there are multiple rods, we might want to separate them or plot all.
    # For now, I'll just plot all points. If it's a single rod simulation, it's fine.
    
    # Check distinct rods
    unique_rods = sorted(list(set(data["rod"])))
    
    plt.figure(figsize=(10, 6))
    
    for r in unique_rods:
        # Filter data for this rod
        rod_indices = [i for i, x in enumerate(data["rod"]) if x == r]
        rod_frames = [data["frame"][i] for i in rod_indices]
        rod_v_sq = [data["v_sq"][i] for i in rod_indices]
        
        plt.plot(rod_frames, rod_v_sq, label=f"Rod {r}", alpha=0.7)

    plt.xlabel("Frame")
    plt.ylabel("Velocity Squared ($v^2$)")
    plt.title(f"Velocity Squared over Time ({csv_path.name})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    
    output_path = csv_path.parent / f"{csv_path.stem}_velocity_squared.png"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
