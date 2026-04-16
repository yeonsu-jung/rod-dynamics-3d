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
            row = df.iloc[[-1]].copy()
            # Try to grab the initial thermal velocity from the first row of each file
            first_row = df.iloc[[0]]
            row['v_0'] = first_row['v0_lin'].values[0]
            dfs.append(row)
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df['gap'] = df['R_cyl'] - (df['d_rod'] / 2.0)
df['gap_over_mu'] = df['gap'] / df['mu']

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
mus = sorted(df['mu'].unique())
colors = plt.cm.viridis(np.linspace(0, 0.9, len(mus)))

for mu, color in zip(mus, colors):
    subset = df[(df['mu'] == mu) & (df['sim_time'] < 1950)] # Only plot those that finished properly
    if len(subset) == 0: continue
    plt.scatter(subset['gap_over_mu'], subset['total_path_length'], label=f'$\mu={mu}$', color=color, alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Gap / $\mu$ (log scale)')
plt.ylabel('Sliding Path Length (finished only)')
plt.title('Raw Unscaled Path Length')

plt.subplot(1, 2, 2)
for mu, color in zip(mus, colors):
    subset = df[(df['mu'] == mu) & (df['sim_time'] < 1950)]
    if len(subset) == 0: continue
    
    # Since theory suggests path length scales with v0^2
    # And potentially gap/mu 
    # Let's plot path / (v0^2) vs gap/mu
    y = subset['total_path_length'] / (subset['v_0'] ** 2)
    plt.scatter(subset['gap_over_mu'], y, label=f'$\mu={mu}$', color=color, alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Gap / $\mu$ (log scale)')
plt.ylabel('Normalized Path Length $S / v_0^2$')
plt.title('Collapse Check')

plt.legend()
plt.tight_layout()
plt.savefig('results/reptation_collapse.png', dpi=300)
print("Plot saved to results/reptation_collapse.png")
