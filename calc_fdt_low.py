
import math

def calc_params(ar, length, diameter, density, dt, T=1.0, gamma=1.0):
    radius = diameter / 2.0
    # Mass
    vol = math.pi * (radius**2) * length
    mass = density * vol
    
    # Inertia (Transverse)
    inertia = (1.0/12.0) * mass * (length**2)
    
    # FDT Formulas: fSigma = sqrt(6 * gamma * m * kT / dt)
    fSigma = math.sqrt(6 * gamma * mass * T / dt)
    tauMag = math.sqrt(6 * gamma * inertia * T / dt)
    
    return {
        "AR": ar,
        "Mass": mass,
        "Inertia": inertia,
        "gamma": gamma,
        "fSigma": fSigma,
        "tauMag": tauMag
    }

configs = [
    {"ar": 30,  "L": 1.0, "D": 1.0/30.0},
    {"ar": 100, "L": 1.0, "D": 0.01},
    {"ar": 200, "L": 1.0, "D": 0.005},
    {"ar": 300, "L": 1.0, "D": 1.0/300.0}
]

density = 1000.0
dt = 0.00025
gamma = 0.1 # Reduced damping

print(f"{'AR':<5} {'Mass':<10} {'Inertia':<10} {'gamma':<10} {'fSigma':<10} {'tauMag':<10}")
print("-" * 65)

for c in configs:
    res = calc_params(c['ar'], c['L'], c['D'], density, dt, gamma=gamma)
    print(f"{res['AR']:<5} {res['Mass']:<10.6f} {res['Inertia']:<10.6f} {res['gamma']:<10.1f} {res['fSigma']:<10.4f} {res['tauMag']:<10.4f}")
