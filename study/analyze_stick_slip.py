import csv
import sys
import math

# Mock plt if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def analyze(filename):
    v0 = []
    v1 = []
    v2 = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['frame'])
                rod_idx = int(row['rod'])
                vx = float(row['vx'])
                vy = float(row['vy'])
                vz = float(row['vz'])
                v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
                
                if rod_idx == 0:
                    v0.append((frame, v_mag))
                elif rod_idx == 1:
                    v1.append((frame, v_mag))
                elif rod_idx == 2:
                    v2.append((frame, v_mag))
                    
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    # Sort by frame
    v0.sort()
    v1.sort()
    v2.sort()
    
    vel0 = [x[1] for x in v0]
    
    f1 = [x[0] for x in v1]
    vel1 = [x[1] for x in v1]
    
    f2 = [x[0] for x in v2]
    vel2 = [x[1] for x in v2]
    
    print(f"Rod 0 (Floor) Max Velocity: {max(vel0) if vel0 else 'N/A'}")
    print(f"Rod 1 (Static Test) Final Velocity: {vel1[-1] if vel1 else 'N/A'}")
    print(f"Rod 2 (Dynamic Test) Final Velocity: {vel2[-1] if vel2 else 'N/A'}")
    
    # Check if Rod 1 stayed still (allow small numerical noise)
    max_v1 = max(vel1) if vel1 else 0
    if max_v1 < 1e-3:
        print("PASS: Rod 1 stuck as expected.")
    else:
        print(f"FAIL: Rod 1 moved! Max velocity: {max_v1}")
        
    # Check if Rod 2 accelerated
    if vel2 and vel2[-1] > 1.0: # It started at 1.0 (approx) and should accelerate
        print(f"PASS: Rod 2 accelerated. Final v: {vel2[-1]}")
    else:
        print(f"FAIL: Rod 2 did not accelerate significantly. Final v: {vel2[-1] if vel2 else 0}")

    # Plot
    if HAS_MATPLOTLIB:
        plt.figure()
        plt.plot(f1, vel1, label='Rod 1 (Should Stick)')
        plt.plot(f2, vel2, label='Rod 2 (Should Slide)')
        plt.xlabel('Frame')
        plt.ylabel('Velocity Magnitude')
        plt.title('Stick-Slip Test')
        plt.legend()
        plt.savefig('stick_slip_result.png')
        print("Plot saved to stick_slip_result.png")
    else:
        print("Matplotlib not found, skipping plot.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze_stick_slip.py <csv_file>")
