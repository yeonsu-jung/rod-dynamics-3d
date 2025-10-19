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
import scipy.optimize as opt

# Base scene file
BASE_SCENE = "../assets/scenes/dissipation_study_sample.json"
OUTPUT_DIR = "scenes"
CSV_DIR = "csvs"
STEPS = 5000  # Shorter for study

# Rod counts to test
# COUNTS = [1000]

# Aspect ratios (length / diameter)
ASPECTS = [25, 50, 100, 200, 500]

# Domain volume
VOLUME = 3.0 ** 3  # [-1.5,1.5]^3
LENGTH = 1.0

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_scenes():
    base = load_json(BASE_SCENE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = {}
    for aspect in ASPECTS:
        diameter = LENGTH / aspect
        count = int(VOLUME / (diameter * LENGTH**2))
        counts[aspect] = count
        scene = base.copy()
        scene['scene']['populate']['count'] = count
        scene['scene']['bodies'][0]['diameter'] = diameter
        filename = f"dissipation_{count}_{aspect}.json"
        save_json(scene, os.path.join(OUTPUT_DIR, filename))
        print(f"Generated {filename} with {count} rods")
    return counts

def run_simulations(counts):
    os.makedirs(CSV_DIR, exist_ok=True)
    for aspect in ASPECTS:
        count = counts[aspect]
        scene_file = os.path.join(OUTPUT_DIR, f"dissipation_{count}_{aspect}.json")
        csv_file = os.path.join(CSV_DIR, f"dissipation_{count}_{aspect}.csv")
        if os.path.exists(csv_file):
            print(f"CSV {csv_file} already exists, skipping simulation.")
            continue
        cmd = [
            "../build/rigidbody_viewer_3d",
            "--headless",
            "--scene", scene_file,
            "--steps", str(STEPS),
            "--csv", csv_file
        ]
        print(f"Running simulation for {count} rods, aspect {aspect}...")
        subprocess.run(cmd, check=True)

def analyze_dissipation(counts):
    results = {}
    for aspect in ASPECTS:
        count = counts[aspect]
        csv_file = os.path.join(CSV_DIR, f"dissipation_{count}_{aspect}.csv")
        df = pd.read_csv(csv_file)
        ke = df['KE'].values
        frames = df['frame'].values
        
        # Fit exponential decay: KE(t) = a * exp(b * t)
        def exp_decay(t, a, b):
            return a * np.exp(b * t)
        
        try:
            popt, pcov = opt.curve_fit(exp_decay, frames, ke, p0=[ke[0], -0.001])
            a, b = popt
            decay_exponent = b
        except:
            print(f"Exponential fit failed for aspect {aspect}")
            decay_exponent = np.nan
        
        # Fit power law: log(KE) = log(c) - p * log(t), so KE ~ t^{-p}
        try:
            # Use last 1/4 of data points for fitting
            n = len(frames)
            start_idx = (n * 3) // 4  # Start from 3/4 of the way through
            t_fit = frames[start_idx:]
            ke_fit = ke[start_idx:]
            log_t = np.log(t_fit)
            log_ke = np.log(ke_fit)
            slope, intercept = np.polyfit(log_t, log_ke, 1)
            power_law_exponent = -slope  # p in KE ~ t^{-p}
        except:
            print(f"Power law fit failed for aspect {aspect}")
            power_law_exponent = np.nan
        
        results[aspect] = {
            'count': count,
            'initial_KE': ke[0],
            'final_KE': ke[-1],
            'decay_exponent': decay_exponent,
            'power_law_exponent': power_law_exponent,
            'frames': frames,
            'ke': ke
        }
        print(f"Aspect {aspect}, Count {count}: Initial KE {ke[0]:.3f}, Final KE {ke[-1]:.3f}, Exp b {decay_exponent:.6f}, Power p {power_law_exponent:.6f}")

    # Plot KE decay with exponential fitted curves
    plt.figure(figsize=(10, 6))
    for aspect in ASPECTS:
        count = counts[aspect]
        label = f'{count} rods, AR {aspect}'
        plt.plot(results[aspect]['frames'], results[aspect]['ke'], label=label)
        # Plot exponential fitted curve
        if not np.isnan(results[aspect]['decay_exponent']):
            a = results[aspect]['initial_KE']  # approx
            b = results[aspect]['decay_exponent']
            fitted_ke_exp = a * np.exp(b * results[aspect]['frames'])
            plt.plot(results[aspect]['frames'], fitted_ke_exp, '--', label=f'Exp fit AR {aspect}')
    plt.xlabel('Frame')
    plt.ylabel('Kinetic Energy (J)')
    plt.legend()
    plt.title('Kinetic Energy Decay with Exponential Fits')
    plt.savefig('ke_decay_exponential_fits.png')
    plt.show()

    # Plot KE decay with power law fitted curves
    plt.figure(figsize=(10, 6))
    for aspect in ASPECTS:
        count = counts[aspect]
        label = f'{count} rods, AR {aspect}'
        plt.plot(results[aspect]['frames'], results[aspect]['ke'], label=label)
        # Plot power law fitted curve (from frame 1 onwards)
        if not np.isnan(results[aspect]['power_law_exponent']):
            t_plot = results[aspect]['frames'][1:]
            ke_plot = results[aspect]['ke'][1:]
            log_t = np.log(t_plot)
            log_ke = np.log(ke_plot)
            slope, intercept = np.polyfit(log_t, log_ke, 1)
            c = np.exp(intercept)
            p = -slope
            fitted_ke_power = c / (t_plot ** p)
            plt.plot(t_plot, fitted_ke_power, ':', label=f'Power fit AR {aspect}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frame')
    plt.ylabel('Kinetic Energy (J)')
    plt.legend()
    plt.title('Kinetic Energy Decay with Power Law Fits')
    plt.savefig('ke_decay_power_fits.png')
    plt.show()

    # Plot exponential decay exponents vs aspect ratio
    aspects = ASPECTS
    exp_exponents = [results[a]['decay_exponent'] for a in aspects]
    plt.figure()
    plt.plot(aspects, exp_exponents, 'o-')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Exponential Decay Exponent b')
    plt.title('Exponential Decay Exponent vs Aspect Ratio')
    plt.savefig('exponential_exponents.png')
    plt.show()

    # Plot power law exponents vs aspect ratio
    power_exponents = [results[a]['power_law_exponent'] for a in aspects]
    plt.figure()
    plt.plot(aspects, power_exponents, 's-')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Power Law Exponent p')
    plt.title('Power Law Exponent vs Aspect Ratio')
    plt.savefig('power_exponents.png')
    plt.show()

if __name__ == "__main__":
    counts = generate_scenes()
    run_simulations(counts)
    analyze_dissipation(counts)
