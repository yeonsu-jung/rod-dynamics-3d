import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

out_dir = "results/reptation"
files = glob.glob(os.path.join(out_dir, "rept_*.csv"))

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df.iloc[[-1]])
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df['gap'] = df['R_cyl'] - (df['d_rod'] / 2.0)
df['gap_over_mu'] = df['gap'] / df['mu']

plt.figure(figsize=(8, 6))

mus = sorted(df['mu'].unique())
colors = plt.cm.viridis(np.linspace(0, 0.9, len(mus)))

for mu, color in zip(mus, colors):
    subset = df[df['mu'] == mu]
    grouped = subset.groupby('gap_over_mu').agg({
        'total_path_length': ['mean', 'std']
    }).reset_index()
    
    x = grouped['gap_over_mu']
    y = grouped['total_path_length']['mean']
    yerr = grouped['total_path_length']['std']
    
    plt.errorbar(x, y, yerr=yerr, fmt='o-', label=f'$\mu={mu}$', color=color, capsize=4, capthick=1.5, alpha=0.8)

plt.axhline(y=1500, color='r', linestyle='--', alpha=0.5, label='Time Limit Boundary')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Gap / $\mu$ (log scale)')
plt.ylabel('Mean Sliding Path Length (log scale)')
plt.title('Frictional Reptation Sliding Path vs Gap/$\mu$')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('results/reptation_path_length_vs_gap_mu.png', dpi=300)
print("Plot saved to results/reptation_path_length_vs_gap_mu.png")
