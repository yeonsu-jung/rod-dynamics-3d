import pandas as pd
import glob
import os

out_dir = "results/reptation"
files = glob.glob(os.path.join(out_dir, "rept_*.csv"))

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df.iloc[[-1]])  # Only take the last row of each simulated trial
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df['gap'] = df['R_cyl'] - (df['d_rod'] / 2.0)
df['gap_over_mu'] = df['gap'] / df['mu']

res = df.groupby(['mu', 'R_cyl']).agg({
    'total_path_length': ['mean', 'std'],
    'sim_time': ['mean', 'max'],
    'gap_over_mu': 'mean'
}).reset_index()

print(res.to_string(index=False))

res2 = df.groupby('gap_over_mu').agg({
    'total_path_length': ['mean', 'std'],
    'sim_time': ['mean']
}).reset_index()

print("\n--- By gap/mu ---")
print(res2.to_string(index=False))
