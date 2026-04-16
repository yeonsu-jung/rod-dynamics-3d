#!/usr/bin/env python3
import argparse
import subprocess
import os
import pandas as pd
from pathlib import Path

ROOT_NAME = 'rod-dynamics-3d'

def find_root_dir(name=ROOT_NAME):
    p = Path(__file__).resolve()
    # check file path and its parents
    for d in [p] + list(p.parents):
        if d.name == name:
            return d
    # fallback: check current working directory and its parents
    cwd = Path.cwd().resolve()
    for d in [cwd] + list(cwd.parents):
        if d.name == name:
            return d
    raise RuntimeError(f"project root '{name}' not found")

ROOT_DIR = str(find_root_dir())

BIN = os.path.join('build', 'rigidbody_viewer_3d')

def run_scene(scene, steps, csv):
    cmd = [BIN, '--scene', scene, '--headless', '--steps', str(steps), '--csv', csv]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def load_ke(csv):
    df = pd.read_csv(csv)
    cols = [c for c in df.columns if c.startswith('KE')]
    return df[['frame', 'KE'] + [c for c in cols if c != 'KE']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-a', default='assets/scenes/pbc_500_rods_elastic.json')
    ap.add_argument('--scene-b', default='assets/scenes/pbc_500_rods_elastic_b0.json')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--csv-a', default='profile_a.csv')
    ap.add_argument('--csv-b', default='profile_b.csv')
    args = ap.parse_args()

    run_scene(args.scene_a, args.steps, args.csv_a)
    run_scene(args.scene_b, args.steps, args.csv_b)

    a = load_ke(args.csv_a)
    b = load_ke(args.csv_b)

    print('\nA last KE:', a['KE'].iloc[-1])
    print('B last KE:', b['KE'].iloc[-1])
    if 'KE_after_posCorrect' in a.columns:
        print('A posCorrect drop (first-last):', a['KE_after_posCorrect'].iloc[0] - a['KE_after_posCorrect'].iloc[-1])
    if 'KE_after_posCorrect' in b.columns:
        print('B posCorrect drop (first-last):', b['KE_after_posCorrect'].iloc[0] - b['KE_after_posCorrect'].iloc[-1])

    # print simple diffs
    print('\nFrame, KE_A, KE_B, dKE')
    for i in range(0, min(len(a), len(b), 50)):
        print(int(a['frame'].iloc[i]), a['KE'].iloc[i], b['KE'].iloc[i], b['KE'].iloc[i]-a['KE'].iloc[i])

if __name__ == '__main__':
    main()
