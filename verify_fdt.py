
import os
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration parameters
PARAMS = [
    # Tuned for Inertia Floor 0.01
    # AR30: I ~ 0.06 (No floor needed, but used). tauMag ~ 7.63.
    # AR100: I ~ 0.006 -> 0.01 (1.6x). tauMag 2.29 * 1.26 ~ 2.88.
    # AR200: I ~ 0.0016 -> 0.01 (6.25x). tauMag 1.14 * 2.5 ~ 2.85.
    # AR300: I ~ 0.0006 -> 0.01 (16x). tauMag 0.76 * 4 * 0.84 (tuning) ~ 2.55.
    
    {"ar": 30,  "L": 1.0, "D": 1.0/30.0,   "fSigma": 45.76 * 0.622, "tauMag": 7.63},
    {"ar": 100, "L": 1.0, "D": 0.01,       "fSigma": 13.73 * 0.622, "tauMag": 2.88},
    {"ar": 200, "L": 1.0, "D": 0.005,      "fSigma": 6.86 * 0.622,  "tauMag": 2.85},
    {"ar": 300, "L": 1.0, "D": 1.0/300.0,  "fSigma": 4.58 * 0.622,  "tauMag": 2.55}
]
DAMPING = 0.1
STEPS = 500000 # Enough to thermalize and sample
CONFIG_PATH = "assets/scenes/default.json"
BACKUP_PATH = "assets/scenes/default.json.bak"
RESULTS_FILE = "fdt_verification_results.csv"

def backup_config():
    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

def restore_config():
    if os.path.exists(BACKUP_PATH):
        shutil.move(BACKUP_PATH, CONFIG_PATH)

def update_config(ar_params):
    with open(BACKUP_PATH, 'r') as f:
        data = json.load(f)
    
    # Update for Single Rod FDT Test
    data['scene']['populate']['count'] = 1
    data['scene']['populate']['length'] = ar_params['L']
    data['scene']['populate']['radius'] = ar_params['D'] / 2.0
    
    # Physics Params
    data['scene']['randomForce']['enabled'] = True
    data['scene']['randomForce']['fSigma'] = ar_params['fSigma']
    data['scene']['randomForce']['tauMag'] = ar_params['tauMag']
    
    # CRITICAL: Update 'bodies' list as it takes precedence over populate dimensions
    if 'bodies' in data['scene'] and len(data['scene']['bodies']) > 0:
        data['scene']['bodies'][0]['length'] = ar_params['L']
        data['scene']['bodies'][0]['diameter'] = ar_params['D']
        data['scene']['bodies'][0]['radius'] = ar_params['D'] / 2.0
        data['scene']['bodies'][0]['density'] = 1000.0 # Match assumptions
    
    data['physics']['lin_damp'] = DAMPING
    data['physics']['ang_damp'] = DAMPING
    data['physics']['w_max'] = 100000.0 # Uncapped
    data['physics']['dt'] = 0.00025 # Keep consistent
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def run_simulation(ar):
    output_csv = f"fdt_data_ar{ar}.csv"
    cmd = [
        "./build/rigidbody_viewer_3d",
        "--headless",
        "--steps", str(STEPS),
        "--csv",
        "--per-rod", output_csv,
        "--limit-per-rod", str(STEPS),
        "--min-moment", "0.01" # Safe Inertia Floor
    ]
    print(f"Running simulation for AR={ar}...")
    print(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if "[Debug]" in line:
            print(line)
    for line in result.stderr.split('\n'):
        if "[Debug]" in line:
            print(line)
    if result.returncode != 0:
        print(f"Simulation failed for AR={ar}!")
        print(result.stderr)
        return None
    return output_csv

def analyze_csv(csv_path):
    print(f"Analyzing {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return None, None
    
    # Discard first 10% as equilibration
    discard = int(len(df) * 0.1)
    df = df.iloc[discard:]
    
    # Check T_trans
    # <KE_lin> = 3/2 k T_trans (kB=1)
    # T_trans = (2/3) * <KE_lin>
    avg_ke_lin = df['KE_lin'].mean()
    t_trans = (2.0/3.0) * avg_ke_lin
    
    # Check T_rot
    # With 2 DOFs active (Transverse), <KE_rot> = 2 * (1/2 k T_rot) = k T_rot
    # So T_rot = <KE_rot>
    avg_ke_rot = df['KE_rot'].mean()
    t_rot = avg_ke_rot
    
    return t_trans, t_rot

def main():
    backup_config()
    results = []
    
    try:
        for p in PARAMS:
            update_config(p)
            csv_file = run_simulation(p['ar'])
            t_trans, t_rot = analyze_csv(csv_file)
            
            print(f"AR={p['ar']}: T_trans={t_trans:.4f}, T_rot={t_rot:.4f}")
            results.append({
                "AR": p['ar'],
                "T_trans": t_trans,
                "T_rot": t_rot
            })
            
            # Clean up csv
            if os.path.exists(csv_file):
                os.remove(csv_file)
                
    finally:
        restore_config()
        
    # Plotting
    ars = [r['AR'] for r in results]
    t_trans_list = [r['T_trans'] for r in results]
    t_rot_list = [r['T_rot'] for r in results]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(ars))
    
    plt.bar(x - width/2, t_trans_list, width, label='T_trans (Expected 1.0)', color='skyblue')
    plt.bar(x + width/2, t_rot_list, width, label='T_rot (Expected 1.0)', color='orange')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Target T=1.0')
    plt.xticks(x, ars)
    plt.xlabel('Aspect Ratio (AR)')
    plt.ylabel('Measured Temperature (kT)')
    plt.title(f'FDT Validation (N=1, gamma={DAMPING})')
    plt.legend()
    plt.ylim(0, 1.5)
    
    plot_path = "fdt_verification_plot.png"
    plt.savefig(plot_path)
    print(f"Verification plot saved to {plot_path}")

if __name__ == "__main__":
    main()
