#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    cols = [
        'KE','KE_after_integrate','KE_after_warmstart','KE_after_solve','KE_after_posCorrect','KE_after_pbcWrap'
    ]
    have = [c for c in cols if c in df.columns]
    if not have:
        print('No KE stage columns found.')
        return
    x = df['frame'] if 'frame' in df.columns else range(len(df))
    plt.figure(figsize=(10,6))
    for c in have:
        plt.plot(x, df[c], label=c)
    if 'jn_sum' in df.columns:
        ax2 = plt.twinx()
        ax2.plot(x, df['jn_sum'], 'r:', alpha=0.5, label='jn_sum')
        ax2.set_ylabel('jn_sum')
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('KE')
    plt.title('Kinetic Energy per Stage')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
