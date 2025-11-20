#!/usr/bin/env python3
"""
analyze_contact_network.py

Helper script to analyze contact network data from rod dynamics simulations.

The network.csv file contains:
- frame: simulation frame number
- rod_i, rod_j: indices of rods in contact (contact hash)
- contact_x, contact_y, contact_z: spatial location of contact point
- normal_x, normal_y, normal_z: contact normal vector (defines contact frame)
- distance: contact penetration distance

This script demonstrates how to:
1. Extract contact pairs (network topology)
2. Analyze contact persistence over time
3. Visualize contact locations and orientations
4. Compute tangent vectors for contact frame

Usage:
    python analyze_contact_network.py --network network.csv

For computing contact forces, you would need to also export force data
from the simulation (not currently implemented in network.csv).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compute_tangent_vectors(normal):
    """
    Given a contact normal vector, compute two orthogonal tangent vectors
    to form a complete contact frame (n, t1, t2).
    
    Args:
        normal: (3,) array of contact normal vector
        
    Returns:
        t1, t2: two orthogonal tangent vectors
    """
    n = normal / np.linalg.norm(normal)
    
    # Choose an arbitrary vector not parallel to n
    if abs(n[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    
    # First tangent via cross product
    t1 = np.cross(n, v)
    t1 = t1 / np.linalg.norm(t1)
    
    # Second tangent orthogonal to both
    t2 = np.cross(n, t1)
    t2 = t2 / np.linalg.norm(t2)
    
    return t1, t2


def analyze_network(network_path):
    """Analyze contact network data."""
    
    df = pd.read_csv(network_path)
    print(f"Loaded {len(df)} contact events from {network_path}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Basic statistics
    print("=" * 60)
    print("CONTACT NETWORK ANALYSIS")
    print("=" * 60)
    
    num_frames = df['frame'].nunique()
    print(f"Number of frames: {num_frames}")
    print(f"Total contact events: {len(df)}")
    print(f"Contacts per frame: {len(df) / num_frames:.2f} (mean)")
    print()
    
    # Create contact pairs
    df['pair'] = df.apply(lambda row: tuple(sorted([row['rod_i'], row['rod_j']])), axis=1)
    unique_pairs = df['pair'].nunique()
    print(f"Unique rod pairs in contact: {unique_pairs}")
    print()
    
    # Contact persistence
    pair_counts = df.groupby('pair').size().sort_values(ascending=False)
    print("Most persistent contacts:")
    for i, (pair, count) in enumerate(pair_counts.head(10).items(), 1):
        persistence = count / num_frames * 100
        print(f"  {i}. Rods {int(pair[0])}-{int(pair[1])}: {count} frames ({persistence:.1f}%)")
    print()
    
    # Contact spatial distribution
    if all(col in df.columns for col in ['contact_x', 'contact_y', 'contact_z']):
        print("Contact spatial distribution:")
        print(f"  X range: [{df['contact_x'].min():.3f}, {df['contact_x'].max():.3f}]")
        print(f"  Y range: [{df['contact_y'].min():.3f}, {df['contact_y'].max():.3f}]")
        print(f"  Z range: [{df['contact_z'].min():.3f}, {df['contact_z'].max():.3f}]")
        print()
    
    # Contact normal distribution
    if all(col in df.columns for col in ['normal_x', 'normal_y', 'normal_z']):
        normals = df[['normal_x', 'normal_y', 'normal_z']].values
        norm_mags = np.linalg.norm(normals, axis=1)
        print(f"Contact normal magnitudes: mean={norm_mags.mean():.3f}, std={norm_mags.std():.3e}")
        
        # Most common normal directions (cluster analysis could be added here)
        print()
    
    # Distance statistics
    if 'distance' in df.columns:
        print("Contact penetration distances:")
        print(f"  Mean: {df['distance'].mean():.3e}")
        print(f"  Std:  {df['distance'].std():.3e}")
        print(f"  Max:  {df['distance'].max():.3e}")
        print()
    
    # Force statistics (if available)
    force_cols = ['force_a_x', 'force_a_y', 'force_a_z']
    if all(col in df.columns for col in force_cols):
        print("=" * 60)
        print("FORCE ANALYSIS")
        print("=" * 60)
        
        # Compute force magnitudes
        df['force_a_mag'] = np.sqrt(
            df['force_a_x']**2 + df['force_a_y']**2 + df['force_a_z']**2
        )
        df['force_b_mag'] = np.sqrt(
            df['force_b_x']**2 + df['force_b_y']**2 + df['force_b_z']**2
        )
        
        print("Normal force magnitudes:")
        print(f"  Mean: {df['force_a_mag'].mean():.3e}")
        print(f"  Std:  {df['force_a_mag'].std():.3e}")
        print(f"  Max:  {df['force_a_mag'].max():.3e}")
        print()
        
        # Friction statistics (if available)
        friction_cols = ['friction_a_x', 'friction_a_y', 'friction_a_z']
        if all(col in df.columns for col in friction_cols):
            df['friction_a_mag'] = np.sqrt(
                df['friction_a_x']**2 + df['friction_a_y']**2 + df['friction_a_z']**2
            )
            
            print("Friction force magnitudes:")
            print(f"  Mean: {df['friction_a_mag'].mean():.3e}")
            print(f"  Std:  {df['friction_a_mag'].std():.3e}")
            print(f"  Max:  {df['friction_a_mag'].max():.3e}")
            
            # Friction to normal force ratio
            df['friction_ratio'] = df['friction_a_mag'] / (df['force_a_mag'] + 1e-12)
            print(f"\nFriction/Normal force ratio:")
            print(f"  Mean: {df['friction_ratio'].mean():.3e}")
            print(f"  Max:  {df['friction_ratio'].max():.3e}")
            print()
        
        # Verify Newton's third law (force_a = -force_b)
        force_diff = np.sqrt(
            (df['force_a_x'] + df['force_b_x'])**2 +
            (df['force_a_y'] + df['force_b_y'])**2 +
            (df['force_a_z'] + df['force_b_z'])**2
        )
        print("Newton's 3rd law verification (force_a + force_b should be ~0):")
        print(f"  Mean error: {force_diff.mean():.3e}")
        print(f"  Max error:  {force_diff.max():.3e}")
        print()
    
    # Example: compute tangent vectors and forces in contact frame for first contact
    if all(col in df.columns for col in ['normal_x', 'normal_y', 'normal_z']) and len(df) > 0:
        print("=" * 60)
        print("CONTACT FRAME EXAMPLE")
        print("=" * 60)
        first_contact = df.iloc[0]
        normal = np.array([first_contact['normal_x'], 
                          first_contact['normal_y'], 
                          first_contact['normal_z']])
        t1, t2 = compute_tangent_vectors(normal)
        
        print(f"Contact: Rods {int(first_contact['rod_i'])}-{int(first_contact['rod_j'])} at frame {int(first_contact['frame'])}")
        print(f"Position: ({first_contact['contact_x']:.3f}, {first_contact['contact_y']:.3f}, {first_contact['contact_z']:.3f})")
        print(f"Normal:   ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
        print(f"Tangent1: ({t1[0]:.3f}, {t1[1]:.3f}, {t1[2]:.3f})")
        print(f"Tangent2: ({t2[0]:.3f}, {t2[1]:.3f}, {t2[2]:.3f})")
        
        # Verify orthogonality
        print(f"\nOrthogonality check:")
        print(f"  n · t1 = {np.dot(normal/np.linalg.norm(normal), t1):.3e}")
        print(f"  n · t2 = {np.dot(normal/np.linalg.norm(normal), t2):.3e}")
        print(f"  t1 · t2 = {np.dot(t1, t2):.3e}")
        
        # Project forces into contact frame if available
        if all(col in df.columns for col in ['force_a_x', 'force_a_y', 'force_a_z']):
            force_a = np.array([first_contact['force_a_x'],
                               first_contact['force_a_y'],
                               first_contact['force_a_z']])
            
            n_unit = normal / np.linalg.norm(normal)
            f_n = np.dot(force_a, n_unit)
            f_t1 = np.dot(force_a, t1)
            f_t2 = np.dot(force_a, t2)
            
            print(f"\nForce on rod A in contact frame:")
            print(f"  Normal component (fN):  {f_n:.6f}")
            print(f"  Tangent1 component (fT1): {f_t1:.6f}")
            print(f"  Tangent2 component (fT2): {f_t2:.6f}")
            
            if all(col in df.columns for col in ['friction_a_x', 'friction_a_y', 'friction_a_z']):
                friction_a = np.array([first_contact['friction_a_x'],
                                      first_contact['friction_a_y'],
                                      first_contact['friction_a_z']])
                fric_mag = np.linalg.norm(friction_a)
                print(f"\nFriction force magnitude: {fric_mag:.6e}")
        
        print()
    
    print("=" * 60)
    
    force_cols = ['force_a_x', 'force_a_y', 'force_a_z']
    if not all(col in df.columns for col in force_cols):
        print("\nNote: Force data not available in network.csv")
        print("      This occurs when using hard contacts or MuJoCo contacts.")
        print("      Force data is only exported for standard soft contacts.")
    
    print("=" * 60)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze contact network data")
    parser.add_argument('--network', type=str, required=True, 
                       help='Path to network.csv file')
    args = parser.parse_args()
    
    network_path = Path(args.network)
    if not network_path.exists():
        print(f"Error: File not found: {network_path}")
        return
    
    df = analyze_network(network_path)
    

if __name__ == "__main__":
    main()
