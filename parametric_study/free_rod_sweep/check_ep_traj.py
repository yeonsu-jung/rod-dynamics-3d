import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run the free-rod simulation.")
parser.add_argument("traj_path", type=str, default=" ", help=f"traj_path")
args = parser.parse_args()

# get endpoints trajectory
ep_traj = np.loadtxt(args.traj_path, skiprows=1, delimiter=",", ndmin=2)[:, 3:]

centroids = (ep_traj[:,:3] + ep_traj[:,3:])/2

dumb_path_length = np.linalg.norm(centroids[-1] - centroids[0]) # todo: get proper sliding length
print(dumb_path_length)