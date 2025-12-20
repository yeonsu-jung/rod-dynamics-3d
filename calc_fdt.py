
import math

def calc_params(ar, length, diameter, density, dt, T=1.0, gamma=1.0):
    radius = diameter / 2.0
    # Mass (Approx cylinder)
    vol = math.pi * (radius**2) * length
    mass = density * vol
    
    # Inertia (Transverse rod approx: 1/12 m L^2)
    inertia = (1.0/12.0) * mass * (length**2)
    
    # FDT Formulas derived:
    # fSigma = sqrt(6 * gamma * m * kB * T / dt)
    # tauMag = sqrt(6 * gamma * I * kB * T / dt)
    # Assuming kB=1
    
    fSigma = math.sqrt(6 * gamma * mass * T / dt)
    tauMag = math.sqrt(6 * gamma * inertia * T / dt)
    
    return {
        "AR": ar,
        "Mass": mass,
        "Inertia": inertia,
        "lin_damp": gamma,
        "ang_damp": gamma,
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
gamma = 1.0 # Suggesting standard damping

print(f"{'AR':<5} {'Mass':<10} {'Inertia':<10} {'lin_damp':<10} {'ang_damp':<10} {'fSigma':<10} {'tauMag':<10}")
print("-" * 70)

for c in configs:
    res = calc_params(c['ar'], c['L'], c['D'], density, dt, gamma=gamma)
    print(f"{res['AR']:<5} {res['Mass']:<10.6f} {res['Inertia']:<10.6f} {res['lin_damp']:<10.1f} {res['ang_damp']:<10.1f} {res['fSigma']:<10.4f} {res['tauMag']:<10.4f}")
