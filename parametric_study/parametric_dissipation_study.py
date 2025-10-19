#!/usr/bin/env python3
"""
parametric_dissipation_study.py

Generates scene files with varying rod counts, runs simulations, and analyzes dissipation rates.
"""

import json
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base scene file
BASE_SCENE = "../assets/scenes/dissipation_study_sample.json"
OUTPUT_DIR = "scenes"
CSV_DIR = "csvs"
STEPS = 5000  # Shorter for study

# Rod counts to test
COUNTS = [1000]

# Aspect ratios (length / diameter)
ASPECTS = [25, 50, 100, 200, 500]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_scenes():
    base = load_json(BASE_SCENE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for count in COUNTS:
        for aspect in ASPECTS:
            scene = base.copy()
            scene['scene']['populate']['count'] = count
            scene['scene']['bodies'][0]['diameter'] = 1.0 / aspect
            filename = f"dissipation_{count}_{aspect}.json"
            save_json(scene, os.path.join(OUTPUT_DIR, filename))
            print(f"Generated {filename}")

def run_simulations():
    os.makedirs(CSV_DIR, exist_ok=True)
    for count in COUNTS:
        for aspect in ASPECTS:
            scene_file = os.path.join(OUTPUT_DIR, f"dissipation_{count}_{aspect}.json")
            csv_file = os.path.join(CSV_DIR, f"dissipation_{count}_{aspect}.csv")
            cmd = [
                "../build/rigidbody_viewer_3d",
                "--headless",
                "--scene", scene_file,
                "--steps", str(STEPS),
                "--csv", csv_file
            ]
            print(f"Running simulation for {count} rods, aspect {aspect}...")
            subprocess.run(cmd, check=True)

def analyze_dissipation():
    results = {}
    for count in COUNTS:
        for aspect in ASPECTS:
            csv_file = os.path.join(CSV_DIR, f"dissipation_{count}_{aspect}.csv")
            df = pd.read_csv(csv_file)
            ke = df['KE'].values
            frames = df['frame'].values
            # Compute dissipation rate as average dKE/dt (negative)
            dke = np.diff(ke)
            dt = np.diff(frames)  # Assuming dt=1 per frame
            rate = np.mean(dke / dt)
            results[(count, aspect)] = {
                'initial_KE': ke[0],
                'final_KE': ke[-1],
                'dissipation_rate': rate,  # J per frame
                'frames': frames,
                'ke': ke
            }
            print(f"Count {count}, Aspect {aspect}: Initial KE {ke[0]:.3f}, Final KE {ke[-1]:.3f}, Rate {rate:.6f} J/frame")

    # Plot
    plt.figure(figsize=(10, 6))
    for count in COUNTS:
        for aspect in ASPECTS:
            label = f'{count} rods, AR {aspect}'
            plt.plot(results[(count, aspect)]['frames'], results[(count, aspect)]['ke'], label=label)
    plt.xlabel('Frame')
    plt.ylabel('Kinetic Energy (J)')
    plt.legend()
    plt.title('Kinetic Energy Decay for Different Aspect Ratios')
    plt.savefig('ke_decay.png')
    plt.show()

    # Plot dissipation rate vs aspect
    aspects = ASPECTS
    rates = [results[(COUNTS[0], a)]['dissipation_rate'] for a in aspects]
    plt.figure()
    plt.plot(aspects, rates, 'o-')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Dissipation Rate (J/frame)')
    plt.title('Dissipation Rate vs Aspect Ratio')
    plt.savefig('dissipation_rate.png')
    plt.show()

if __name__ == "__main__":
    generate_scenes()
    run_simulations()
    analyze_dissipation()
