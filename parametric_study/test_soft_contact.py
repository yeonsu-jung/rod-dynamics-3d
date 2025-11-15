#!/usr/bin/env python3
"""
Test soft contact model vs hard impulse solver
Compares energy conservation in 2-rod collision
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os

def generate_two_rod_collision(output_path, use_soft_contact=False):
    """Generate scene with 2 rods colliding head-on"""
    
    # Two rods approaching each other
    rod_length = 0.5
    rod_radius = 0.025
    
    bodies = []
    
    # Rod 1: Moving right at v = +1.0 m/s
    bodies.append({
        "pos": [-0.3, 0, 0],
        "length": rod_length,
        "diameter": 2 * rod_radius,
        "density": 2.0,  # mass = 1 kg
        "rot_axis": [0, 0, 1],  # Rotate around Z
        "rot_deg": 90.0,  # 90° to align rod with X-axis
        "v_lin": [1.0, 0, 0],
        "v_ang": [0, 0, 0]
    })
    
    # Rod 2: Moving left at v = -1.0 m/s
    bodies.append({
        "pos": [0.3, 0, 0],
        "length": rod_length,
        "diameter": 2 * rod_radius,
        "density": 2.0,  # mass = 1 kg
        "rot_axis": [0, 0, 1],  # Rotate around Z  
        "rot_deg": 90.0,  # 90° to align rod with X-axis
        "v_lin": [-1.0, 0, 0],
        "v_ang": [0, 0, 0]
    })
    
    # Physics config
    physics = {
        "gravity": [0, 0, 0],
        "dt": 1.0/12000.0,  # Very small timestep (20x smaller than original)
        "lin_damp": 0.0,
        "ang_damp": 0.0,
        "w_max": 1e6,
        "friction": 0.0,
        "restitution": 1.0,
        "floor": False
    }
    
    if use_soft_contact:
        # Soft contact parameters (tuned for Verlet integration)
        physics["soft_contact"] = {
            "enabled": True,
            "delta": 0.005,  # Transition width
            "k_scaler": 500.0,  # Optimal stiffness (tuned)
            "mu": 0.0,  # No friction for this test
            "nu": 1e-3,
            "enable_friction": False,
            "verbose": False  # Disable verbose for speed
        }
        # Don't need baumgarte correction with soft contacts
        physics["solver"] = {
            "baumgarte": 0.0,
            "velIters": 0
        }
    else:
        # Hard impulse solver (perfect conservation mode)
        physics["soft_contact"] = {
            "enabled": False
        }
        physics["solver"] = {
            "baumgarte": 0.0,
            "allowed_pen": 0.0,
            "velIters": 100,
            "split_impulse": False,
            "split_orient": False
        }
    
    scene = {
        "scene": {
            "bodies": bodies,
            "periodic": {
                "enabled": False
            }
        },
        "physics": physics
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"Generated scene: {output_path}")
    print(f"  Contact model: {'SOFT' if use_soft_contact else 'HARD'}")
    print(f"  Initial KE: {2 * 0.5 * 1.0 * 1.0**2:.3f} J (2 rods, m=1kg, v=1m/s each)")

def run_simulation(scene_path, output_csv, steps=600):
    """Run headless simulation"""
    cmd = [
        "../build/rigidbody_viewer_3d",
        "--scene", scene_path,
        "--headless", str(steps),
        "--perrod", output_csv,
        "--perrod-max", str(steps)
    ]
    
    print(f"\nRunning simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/Users/yeonsu/GitHub/rod-dynamics-3d/parametric_study")
    
    if result.returncode != 0:
        print(f"ERROR: Simulation failed with code {result.returncode}")
        return False
    
    return True

def analyze_collision(csv_path, label):
    """Analyze kinetic energy over time"""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    # CSV columns: frame, rod_id, x, y, z, vx, vy, vz, wx, wy, wz, ...
    frames = []
    ke_total = []
    
    unique_frames = np.unique(data[:, 0])
    
    for frame in unique_frames:
        frame_data = data[data[:, 0] == frame]
        
        ke_frame = 0.0
        for row in frame_data:
            vx, vy, vz = row[5], row[6], row[7]
            wx, wy, wz = row[8], row[9], row[10]
            
            # Assume m = 1 kg, I ≈ m*L²/12 for rod
            mass = 1.0
            L = 0.5
            I = mass * L**2 / 12.0
            
            ke_lin = 0.5 * mass * (vx**2 + vy**2 + vz**2)
            ke_rot = 0.5 * I * (wx**2 + wy**2 + wz**2)
            
            ke_frame += ke_lin + ke_rot
        
        frames.append(frame)
        ke_total.append(ke_frame)
    
    frames = np.array(frames)
    ke_total = np.array(ke_total)
    
    # Compute statistics
    ke_initial = ke_total[0]
    ke_final = ke_total[-1]
    ke_loss = ke_initial - ke_final
    ke_loss_pct = 100 * ke_loss / ke_initial if ke_initial > 0 else 0
    
    print(f"\n{label}:")
    print(f"  Initial KE: {ke_initial:.6f} J")
    print(f"  Final KE:   {ke_final:.6f} J")
    print(f"  Loss:       {ke_loss:.6f} J ({ke_loss_pct:.2f}%)")
    print(f"  Min KE:     {ke_total.min():.6f} J")
    print(f"  Max KE:     {ke_total.max():.6f} J")
    
    return frames, ke_total, ke_loss_pct

def main():
    print("="*60)
    print("SOFT CONTACT vs HARD IMPULSE COMPARISON")
    print("2-Rod Head-On Collision Test")
    print("="*60)
    
    # Generate scenes
    scene_hard = "../assets/scenes/test_collision_hard.json"
    scene_soft = "../assets/scenes/test_collision_soft.json"
    
    generate_two_rod_collision(scene_hard, use_soft_contact=False)
    print()
    generate_two_rod_collision(scene_soft, use_soft_contact=True)
    
    # Run simulations
    csv_hard = "../build/test_collision_hard.csv"
    csv_soft = "../build/test_collision_soft.csv"
    
    steps = 12000  # 1 second at dt=1/12000
    
    print("\n" + "="*60)
    print("RUNNING HARD IMPULSE SOLVER")
    print("="*60)
    success_hard = run_simulation(scene_hard, csv_hard, steps)
    
    print("\n" + "="*60)
    print("RUNNING SOFT CONTACT SOLVER")
    print("="*60)
    success_soft = run_simulation(scene_soft, csv_soft, steps)
    
    if not (success_hard and success_soft):
        print("\nERROR: One or both simulations failed!")
        return
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    frames_hard, ke_hard, loss_hard = analyze_collision(csv_hard, "HARD IMPULSE")
    frames_soft, ke_soft, loss_soft = analyze_collision(csv_soft, "SOFT CONTACT")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time evolution (dt varies by solver)
    dt_hard = 1.0/600.0
    dt_soft = 1.0/12000.0
    time_hard = frames_hard * dt_hard
    time_soft = frames_soft * dt_soft
    
    ax1.plot(time_hard, ke_hard, 'b-', label=f'Hard Impulse (loss: {loss_hard:.1f}%)', linewidth=2)
    ax1.plot(time_soft, ke_soft, 'r-', label=f'Soft Contact (loss: {loss_soft:.1f}%)', linewidth=2)
    ax1.axhline(y=ke_hard[0], color='k', linestyle='--', alpha=0.3, label='Initial KE')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Total Kinetic Energy (J)', fontsize=12)
    ax1.set_title('2-Rod Elastic Collision: Energy Conservation Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Percent deviation from initial
    dev_hard = 100 * (ke_hard - ke_hard[0]) / ke_hard[0]
    dev_soft = 100 * (ke_soft - ke_soft[0]) / ke_soft[0]
    
    ax2.plot(time_hard, dev_hard, 'b-', label='Hard Impulse', linewidth=2)
    ax2.plot(time_soft, dev_soft, 'r-', label='Soft Contact', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Energy Deviation (%)', fontsize=12)
    ax2.set_title('Energy Conservation Error', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_plot = "collision_comparison.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_plot}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    improvement = loss_hard - loss_soft
    print(f"Energy loss reduction: {improvement:.2f} percentage points")
    print(f"  Hard impulse: {loss_hard:.2f}% loss")
    print(f"  Soft contact: {loss_soft:.2f}% loss")
    
    if loss_soft < loss_hard:
        print(f"\n✓ Soft contact preserves energy BETTER by {improvement:.2f}%")
    else:
        print(f"\n✗ Soft contact is WORSE by {-improvement:.2f}%")
        print("  (Try adjusting k_scaler or delta)")

if __name__ == "__main__":
    main()
