#!/usr/bin/env python3
"""analyze_mujoco_batch.py

Analyzes a batch of MuJoCo simulation runs (e.g. "runs_mujoco/mujoco_N500_sweep").
Extracts metrics from `rod_contact_info.csv` (Cluster Size, Contact Number) and
compiles them into `analysis/summary.csv`.

Usage:
  python3 analyze_mujoco_batch.py <batch_dir>
"""

import argparse
import csv
import json
import re
import networkx as nx
import pandas as pd
import math
from pathlib import Path
from tqdm import tqdm
import sys

def parse_run_name(name: str):
    # Parse AR, Friction from name e.g. "2026..._AR100_F0.5_K0.0"
    ar = 0
    fric = 0.0
    
    m_ar = re.search(r"_AR(\d+)", name)
    if m_ar:
        ar = int(m_ar.group(1))
        
    m_f = re.search(r"_F([0-9.]+)", name)
    if m_f:
        fric = float(m_f.group(1))
        
    return ar, fric

def analyze_run(run_dir: Path, n_rods_ref: int) -> dict:
    # Default result
    res = {
        "run_name": run_dir.name,
        "AR": 0,
        "friction": 0,
        "N": n_rods_ref,
        "max_cluster_frac_end": float("nan"),
        "avg_contacts_end": float("nan"),
        "sim_time": 0.0
    }
    
    # Metadata
    ar, fric = parse_run_name(run_dir.name)
    res["AR"] = ar
    res["friction"] = fric
    
    # Load contacts
    contacts_path = run_dir / "rod_contact_info.csv"
    if not contacts_path.exists():
        return res
        
    try:
        # rod_contact_info.csv might be large.
        # Format: step,geom1,geom2,... geom names are "particleX"
        # We only care about the LAST frame/step usually for scaling analysis.
        
        df = pd.read_csv(contacts_path)
        
        if df.empty:
            return res
            
        max_step = df["step"].max()
        final_df = df[df["step"] == max_step]
        
        if final_df.empty:
            return res
            
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n_rods_ref)) # Ensure all N rods are nodes
        
        # Edges
        # geom names are "particleX"
        def geom_to_id(s):
            if isinstance(s, str) and s.startswith("particle"):
                try:
                    return int(s.replace("particle", ""))
                except:
                    return -1
            return -1
            
        for _, row in final_df.iterrows():
            u = geom_to_id(row["geom1"])
            v = geom_to_id(row["geom2"])
            if u >= 0 and v >= 0 and u != v:
                G.add_edge(u, v)
                
        # Metrics
        # Max cluster
        if len(G) > 0:
            largest_cc_size = len(max(nx.connected_components(G), key=len)) if nx.number_of_edges(G) > 0 else 1
            res["max_cluster_frac_end"] = largest_cc_size / n_rods_ref
        
        # Avg contacts (degree)
        avg_deg = sum(dict(G.degree()).values()) / n_rods_ref
        res["avg_contacts_end"] = avg_deg
        
        # read time
        time_path = run_dir / "time_points.txt"
        if time_path.exists():
            with open(time_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Try to parse last line as float
                    try:
                        res["sim_time"] = float(lines[-1].strip())
                    except:
                        pass

    except Exception as e:
        print(f"Error analyzing {run_dir.name}: {e}")
        
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path, help="Batch directory containing runs")
    args = parser.parse_args()
    
    if not args.batch_dir.exists():
        print("Directory not found")
        return
        
    # Infer N from batch dir name if possible (e.g. mujoco_N500_sweep)
    n_match = re.search(r"_N(\d+)_", args.batch_dir.name)
    n_ref = int(n_match.group(1)) if n_match else 0
    
    results = []
    
    print(f"Scanning {args.batch_dir}...")
    run_dirs = sorted([d for d in args.batch_dir.iterdir() if d.is_dir()])
    
    for d in tqdm(run_dirs):
        # Look for output markers
        if (d / "rod_contact_info.csv").exists():
            r_data = analyze_run(d, n_ref)
            if math.isfinite(r_data["max_cluster_frac_end"]):
                results.append(r_data)
                
    if not results:
        print("No results found.")
        return
        
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    
    out_csv = out_dir / "summary.csv"
    
    fieldnames = ["run_name", "AR", "friction", "N", "max_cluster_frac_end", "avg_contacts_end", "sim_time"]
    
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Written summary to {out_csv} ({len(results)} rows)")

if __name__ == "__main__":
    main()
