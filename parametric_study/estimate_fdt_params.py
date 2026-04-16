import numpy as np
import argparse

def calculate_sphere_props(radius, density):
    volume = (4.0/3.0) * np.pi * radius**3
    mass = density * volume
    inertia = 0.4 * mass * radius**2
    return mass, inertia

def calculate_rod_props(length, diameter, density):
    radius = diameter / 2.0
    # Volume of a capsule (cylinder + 2 hemispheres)
    # But the code approximates mass as cylinder? Let's check rigid_body.cpp
    # rigid_body.cpp: volume = pi * r^2 * totalHeight (where totalHeight = 2*halfHeight = length)
    # It ignores the hemispherical caps for mass calculation in the current C++ implementation.
    
    volume = np.pi * radius**2 * length
    mass = density * volume
    
    # Inertia for cylinder axis along Y
    # I_transverse (x, z) = m * (3*r^2 + h^2) / 12
    # I_axial (y) = m * r^2 / 2
    
    i_axial = 0.5 * mass * radius**2
    i_transverse = mass * (3*radius**2 + length**2) / 12.0
    
    return mass, i_transverse, i_axial

def main():
    parser = argparse.ArgumentParser(description="Calculate fSigma and tauMag for FDT.")
    parser.add_argument("--shape", choices=["sphere", "rod"], default="sphere")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=1.0, help="Total length for rod")
    parser.add_argument("--diameter", type=float, default=0.1, help="Diameter for rod")
    parser.add_argument("--density", type=float, default=1000.0)
    
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--temp", type=float, default=1.0, help="Target Temperature (kT)")
    parser.add_argument("--lin_damp", type=float, default=1.0)
    parser.add_argument("--ang_damp", type=float, default=1.0)
    
    args = parser.parse_args()
    
    if args.shape == "sphere":
        mass, inertia = calculate_sphere_props(args.radius, args.density)
        print(f"--- Sphere Properties ---")
        print(f"Radius: {args.radius}")
        print(f"Mass:   {mass:.4f}")
        print(f"Inertia:{inertia:.4f}")
        
        f_sigma = np.sqrt(2 * args.lin_damp * args.temp * mass / args.dt)
        tau_mag = np.sqrt(2 * args.ang_damp * args.temp * inertia / args.dt)
        
        print(f"\n--- Recommended Settings (T={args.temp}) ---")
        print(f"fSigma: {f_sigma:.4f}")
        print(f"tauMag: {tau_mag:.4f}")
        
    elif args.shape == "rod":
        mass, i_trans, i_axial = calculate_rod_props(args.length, args.diameter, args.density)
        print(f"--- Rod (Capsule) Properties ---")
        print(f"Length: {args.length}, Diameter: {args.diameter}")
        print(f"Mass:   {mass:.4f}")
        print(f"Inertia (Transverse): {i_trans:.6f}")
        print(f"Inertia (Axial):      {i_axial:.6f}")
        
        f_sigma = np.sqrt(2 * args.lin_damp * args.temp * mass / args.dt)
        
        # For rods, we usually tune for the transverse mode (tumbling)
        tau_mag_trans = np.sqrt(2 * args.ang_damp * args.temp * i_trans / args.dt)
        tau_mag_axial = np.sqrt(2 * args.ang_damp * args.temp * i_axial / args.dt)
        
        print(f"\n--- Recommended Settings (T={args.temp}) ---")
        print(f"fSigma: {f_sigma:.4f}")
        print(f"tauMag: {tau_mag_trans:.4f}")
        print(f"\nNOTE: The simulation has been updated to apply rotational noise")
        print(f"ONLY to the transverse axes (tumbling) for capsules.")
        print(f"This ensures correct tumbling temperature without exciting the spinning mode.")
        print(f"Use the 'tauMag' value above.")

if __name__ == "__main__":
    main()
