#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Pick one per-rod CSV produced with --perrod
perrod_path = Path("parametric_study/runs/_both_models_demo_patch_test/confined_n20_hard_mu0_10_noise_f1e-05_t1e-05_seed1.csv")
# Pick the aggregate summary CSV
summary_path = Path("parametric_study/runs/_both_models_demo_patch_test/friction_sweep_n20_box0.90_summary.csv")

def read_perrod(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"\nPer-rod file: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)} (frames * rods)")
    # Basic checks
    n_rods = df['rod'].nunique()
    n_frames = df['frame'].nunique()
    print(f"Unique rods: {n_rods} | Unique frames: {n_frames}")
    # Show first frame’s rods
    first_frame = df['frame'].min()
    print(f"\nFirst frame ({first_frame}) sample:")
    print(df[df['frame'] == first_frame].head())
    # Quick KE stats
    print("\nKE_total stats:")
    print(df['KE_total'].describe())

def read_summary(csv_path: Path):
    df = pd.read_csv(csv_path)
    print(f"\nSummary file: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)} (one per (contact_model, μ, fSigma, seed))")
    print("\nHead:")
    print(df.head())
    # Group by contact model for quick average drift
    if 'contact_model' in df.columns:
        drift_grp = df.groupby('contact_model')['drift'].mean()
        print("\nMean drift (%) by contact model:")
        print(drift_grp)

if perrod_path.exists():
    read_perrod(perrod_path)
else:
    print(f"Missing per-rod file: {perrod_path}")

if summary_path.exists():
    read_summary(summary_path)
else:
    print(f"Missing summary file: {summary_path}")
    
