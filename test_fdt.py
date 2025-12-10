import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.optimize import curve_fit

def unwrap_positions(positions, box_size):
    """
    Unwrap periodic boundary conditions for trajectory.
    positions: (N, 3) array of positions
    box_size: float or (3,) array
    """
    # Calculate displacements between consecutive frames
    delta = positions[1:] - positions[:-1]
    
    # If displacement is larger than box_size/2, it's a boundary crossing
    # We subtract (sign(delta) * box_size) to correct it
    crossings = np.abs(delta) > (box_size / 2)
    delta[crossings] -= np.sign(delta[crossings]) * box_size
    
    # Reconstruct unwrapped path
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]
    unwrapped[1:] = positions[0] + np.cumsum(delta, axis=0)
    
    return unwrapped

def exponential_decay(t, A, gamma):
    return A * np.exp(-gamma * t)

def linear_fit(t, D, C):
    return 6 * D * t + C

def analyze_fdt(csv_path, box_size, dt, output_plot=None):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    required_cols = ['frame', 'rod', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'KE_lin', 'KE_rot']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing required columns. Need: {required_cols}")
        return

    # Sort by rod and frame
    df = df.sort_values(['rod', 'frame'])
    
    rods = df['rod'].unique()
    print(f"Found {len(rods)} rods.")
    
    # Storage for accumulated correlation functions
    max_lag = 5000 # frames to calculate correlation for
    vacf_lin_accum = np.zeros(max_lag)
    vacf_rot_accum = np.zeros(max_lag)
    msd_accum = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    # Storage for scalar averages
    avg_ke_lin = 0
    avg_ke_rot = 0
    avg_v2 = 0
    avg_w2 = 0
    total_samples = 0
    
    # Estimate mass and inertia from KE and v^2
    # KE = 0.5 * m * v^2 => m = 2 * KE / v^2
    # We'll collect some samples to estimate this
    mass_estimates = []
    inertia_estimates = [] # This is scalar approximation, I_eff = 2*KE_rot / w^2

    print("Processing trajectories...")
    for rod in rods:
        rod_data = df[df['rod'] == rod]
        
        # Extract arrays
        pos = rod_data[['px', 'py', 'pz']].values
        vel = rod_data[['vx', 'vy', 'vz']].values
        ang_vel = rod_data[['wx', 'wy', 'wz']].values
        ke_lin = rod_data['KE_lin'].values
        ke_rot = rod_data['KE_rot'].values
        
        n_frames = len(rod_data)
        if n_frames < 2:
            continue
            
        # Unwrap positions
        pos_unwrapped = unwrap_positions(pos, box_size)
        
        # Accumulate scalar stats
        avg_ke_lin += np.sum(ke_lin)
        avg_ke_rot += np.sum(ke_rot)
        v2 = np.sum(vel**2, axis=1)
        w2 = np.sum(ang_vel**2, axis=1)
        avg_v2 += np.sum(v2)
        avg_w2 += np.sum(w2)
        total_samples += n_frames
        
        # Mass estimate (avoid div by zero)
        valid_v = v2 > 1e-6
        if np.any(valid_v):
            m_est = 2 * ke_lin[valid_v] / v2[valid_v]
            mass_estimates.extend(m_est)
            
        valid_w = w2 > 1e-6
        if np.any(valid_w):
            i_est = 2 * ke_rot[valid_w] / w2[valid_w]
            inertia_estimates.extend(i_est)

        # Calculate correlations and MSD for lags
        # We use a simple loop for clarity, could be optimized with fft
        # Only go up to max_lag or n_frames
        limit = min(n_frames, max_lag)
        
        for lag in range(limit):
            # VACF Linear: <v(t) . v(0)> -> average over all t0
            # v[t] . v[t+lag]
            v_dot = np.sum(vel[:n_frames-lag] * vel[lag:], axis=1)
            vacf_lin_accum[lag] += np.sum(v_dot)
            
            # VACF Angular
            w_dot = np.sum(ang_vel[:n_frames-lag] * ang_vel[lag:], axis=1)
            vacf_rot_accum[lag] += np.sum(w_dot)
            
            # MSD: <|r(t) - r(0)|^2>
            # |r[t+lag] - r[t]|^2
            disp = pos_unwrapped[lag:] - pos_unwrapped[:n_frames-lag]
            sq_disp = np.sum(disp**2, axis=1)
            msd_accum[lag] += np.sum(sq_disp)
            
            counts[lag] += (n_frames - lag)

    # Normalize
    valid_lags = counts > 0
    vacf_lin = vacf_lin_accum[valid_lags] / counts[valid_lags]
    vacf_rot = vacf_rot_accum[valid_lags] / counts[valid_lags]
    msd = msd_accum[valid_lags] / counts[valid_lags]
    time = np.arange(len(vacf_lin)) * dt
    
    avg_ke_lin /= total_samples
    avg_ke_rot /= total_samples
    avg_v2 /= total_samples
    avg_w2 /= total_samples
    
    est_mass = np.median(mass_estimates) if mass_estimates else 1.0
    est_inertia = np.median(inertia_estimates) if inertia_estimates else 1.0
    
    # Temperature from Equipartition
    # <KE_lin> = 3/2 k_B T_trans
    # <KE_rot> = 3/2 k_B T_rot (if 3 DOFs) or 2/2 k_B T_rot (if 2 DOFs)
    # Assuming k_B = 1
    T_trans = (2/3) * avg_ke_lin
    
    # For rods with only transverse noise, we only excite 2 DOFs.
    # So <KE_rot> = 2 * (1/2 k_B T) = k_B T
    # T_rot = avg_ke_rot
    # But let's keep the standard 3-DOF definition for comparison, 
    # and just note that we expect T_rot to be 2/3 of T_trans if axial mode is cold.
    T_rot = (2/3) * avg_ke_rot
    
    print("\n" + "="*40)
    print("       FDT ANALYSIS REPORT")
    print("="*40)
    print(f"Estimated Mass: {est_mass:.4f}")
    print(f"Estimated Inertia (eff): {est_inertia:.4f}")
    print(f"Temperature (Trans): {T_trans:.6f}")
    print(f"Temperature (Rot):   {T_rot:.6f}")
    print(f"Ratio T_rot/T_trans: {T_rot/T_trans:.4f}")
    print(f"Note: If axial spin is suppressed, expect Ratio ~ 0.66 (2/3)")
    
    # --- Translational Analysis ---
    print("\n--- Translational Motion ---")
    
    # Fit VACF to A * exp(-gamma * t)
    try:
        popt_v, _ = curve_fit(exponential_decay, time, vacf_lin, p0=[vacf_lin[0], 1.0])
        gamma_trans_fit = popt_v[1]
        print(f"VACF Fit Gamma: {gamma_trans_fit:.4f}")
    except:
        print("VACF Fit failed")
        gamma_trans_fit = float('nan')

    # Fit MSD to 6*D*t + C (ignore early ballistic regime)
    # Fit range: e.g., from lag 50 to end
    start_fit = min(50, len(time)//2)
    try:
        popt_m, _ = curve_fit(linear_fit, time[start_fit:], msd[start_fit:])
        D_msd = popt_m[0]
        print(f"MSD Fit Diffusion (D): {D_msd:.6f}")
    except:
        print("MSD Fit failed")
        D_msd = float('nan')
        
    # Check Einstein Relation: D = k_B T / (m gamma)
    # k_B = 1
    if not np.isnan(gamma_trans_fit):
        D_predicted = T_trans / (est_mass * gamma_trans_fit)
        print(f"Predicted D (kT/m*gamma): {D_predicted:.6f}")
        if D_msd > 0:
            print(f"Ratio D_msd / D_pred: {D_msd/D_predicted:.4f} (Expect ~1.0)")
    
    # --- Rotational Analysis ---
    print("\n--- Rotational Motion ---")
    
    # Fit Angular VACF to A * exp(-gamma * t)
    try:
        popt_w, _ = curve_fit(exponential_decay, time, vacf_rot, p0=[vacf_rot[0], 1.0])
        gamma_rot_fit = popt_w[1]
        print(f"Ang VACF Fit Gamma: {gamma_rot_fit:.4f}")
    except:
        print("Ang VACF Fit failed")
        gamma_rot_fit = float('nan')
        
    # Rotational Diffusion D_r = k_B T / (I gamma)
    # We can estimate D_r from integral of VACF_rot?
    # D_r = (1/3) * integral(<w(t).w(0)>) dt
    # For exponential decay: integral = <w^2> / gamma
    # So D_r = <w^2> / (3 gamma)
    
    if not np.isnan(gamma_rot_fit):
        D_rot_vacf = (avg_w2 / 3.0) / gamma_rot_fit
        print(f"Rot Diffusion (from VACF integral): {D_rot_vacf:.6f}")
        
        D_rot_predicted = T_rot / (est_inertia * gamma_rot_fit)
        print(f"Predicted D_rot (kT/I*gamma): {D_rot_predicted:.6f}")
        print(f"Ratio D_obs / D_pred: {D_rot_vacf/D_rot_predicted:.4f} (Expect ~1.0)")

    if output_plot:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(time, vacf_lin, label='Data')
        if not np.isnan(gamma_trans_fit):
            plt.plot(time, exponential_decay(time, *popt_v), '--', label=f'Fit $\gamma={gamma_trans_fit:.2f}$')
        plt.title('Translational VACF')
        plt.xlabel('Time')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(time, msd, label='Data')
        if not np.isnan(D_msd):
            plt.plot(time[start_fit:], linear_fit(time[start_fit:], *popt_m), '--', label=f'Fit $D={D_msd:.4f}$')
        plt.title('Translational MSD')
        plt.xlabel('Time')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(time, vacf_rot, label='Data')
        if not np.isnan(gamma_rot_fit):
            plt.plot(time, exponential_decay(time, *popt_w), '--', label=f'Fit $\gamma={gamma_rot_fit:.2f}$')
        plt.title('Rotational VACF')
        plt.xlabel('Time')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_plot)
        print(f"\nPlots saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FDT on rod simulation data")
    parser.add_argument("csv_file", help="Path to perrod.csv file")
    parser.add_argument("--box", type=float, default=3.0, help="Box size for PBC unwrapping")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    parser.add_argument("--plot", help="Output path for plots (e.g. fdt_plots.png)")
    
    args = parser.parse_args()
    
    analyze_fdt(args.csv_file, args.box, args.dt, args.plot)
