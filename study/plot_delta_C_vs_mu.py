import numpy as np
import matplotlib.pyplot as plt

# Data from analysis steps
# mu values
mus = [0.05, 0.1, 0.15, 0.2, 0.4, 1.0]

# Initial C for all is -8704
C_init = -8704

# Final C values from analysis
# mu=0.05: C_final = 1584 -> Delta C = 10,288
# mu=0.1:  C_final = 1776 -> Delta C = 10,480
# mu=0.15: C_final = -20928 -> Delta C = -12,224
# mu=0.2:  C_final = -5132 -> Delta C = 3,572
# mu=0.4:  C_final = -11004 -> Delta C = -2,300
# mu=1.0:  C_final = -7420 -> Delta C = 1,284

# Absolute change |Delta C|
delta_Cs = [
    abs(1584 - (-8704)),   # 0.05 -> 10288
    abs(1776 - (-8704)),   # 0.1  -> 10480
    abs(-20928 - (-8704)), # 0.15 -> 12224
    abs(-5132 - (-8704)),  # 0.2  -> 3572
    abs(-11004 - (-8704)), # 0.4  -> 2300
    abs(-7420 - (-8704))   # 1.0  -> 1284
]

print("Data:")
for m, dc in zip(mus, delta_Cs):
    print(f"mu={m}, |Delta C|={dc}")

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(mus, delta_Cs, 'o-', linewidth=2, markersize=8)

# Add labels and title
plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
plt.ylabel(r'Topological Change $|\Delta C|$', fontsize=14)
plt.title(r'Topological Stability vs Friction', fontsize=16)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight trend
plt.annotate('High Instability', xy=(0.1, 11000), xytext=(0.2, 11500),
             fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))
             
plt.annotate('Stabilized', xy=(1.0, 1500), xytext=(0.8, 4000),
             fontsize=12, arrowprops=dict(facecolor='black', shrink=0.05))

# Save
plt.savefig('study/delta_C_vs_mu.png', dpi=300)
print("Plot saved to study/delta_C_vs_mu.png")
