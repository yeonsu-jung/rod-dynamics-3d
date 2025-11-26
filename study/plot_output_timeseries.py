import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_output_csv(path: Path):
    frames = []
    contacts = []
    KE = []
    max_overlap = []
    reldisp_norm = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            contacts.append(float(row["contacts"]))
            KE.append(float(row["KE"]))
            max_overlap.append(float(row["max_overlap"]))
            reldisp_norm.append(float(row["reldisp_norm"]))

    return {
        "frame": frames,
        "contacts": contacts,
        "KE": KE,
        "max_overlap": max_overlap,
        "reldisp_norm": reldisp_norm,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_output_timeseries.py path/to/output.csv [dt]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    data = load_output_csv(csv_path)

    # Optional dt argument to convert frames to time
    dt = None
    if len(sys.argv) >= 3:
        try:
            dt = float(sys.argv[2])
        except ValueError:
            print(f"Warning: could not parse dt='{sys.argv[2]}', using frame index instead of time.")
            dt = None

    if dt is not None:
        x = [f * dt for f in data["frame"]]
        x_label = "time [s]"
    else:
        x = data["frame"]
        x_label = "frame"

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(x, data["contacts"])
    axes[0].set_ylabel("contacts")

    axes[1].plot(x, data["KE"])
    axes[1].set_ylabel("KE")

    axes[2].plot(x, data["max_overlap"])
    axes[2].set_ylabel("max_overlap")

    axes[3].plot(x, data["reldisp_norm"])
    axes[3].set_ylabel("reldisp_norm")
    axes[3].set_xlabel(x_label)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
