# create output folder
import datetime
import os
import shutil
import subprocess

# find root folder recursively (mujoco-balls)
from pathlib import Path
# find mujoco-balls in the path of Path(__file__)
root_folder = None
for parent in Path(__file__).parents:
    if parent.name == "mujoco-balls":
        root_folder = parent
        break
    
print(f"Found root folder: {root_folder}")

# New dataset path
base_data_path = '/n/home01/yjung/Github/mujoco-balls/data/relaxation_3rd_multithreading'

file_path_list = []

# Iterate over N folders (N10, N20, ..., N1000)
for n_folder in os.listdir(base_data_path):
    if not n_folder.startswith('N'):
        continue
    
    n_path = os.path.join(base_data_path, n_folder)
    if not os.path.isdir(n_path):
        continue

    # Get seed directories
    seed_dirs = [d for d in os.listdir(n_path) if os.path.isdir(os.path.join(n_path, d))]
    
    if not seed_dirs:
        continue

    # only for N = 20, 200, 1000
    if n_folder not in ['N20', 'N200', 'N1000']:
        continue
    
    # "For now, we only deal with only one random seed data"
    # picking the first one available
    seed_dir = seed_dirs[0]
    seed_path = os.path.join(n_path, seed_dir)
    
    # Get all x_relaxed_AR*.txt files
    for f in os.listdir(seed_path):
        if f.startswith('x_relaxed_AR') and f.endswith('.txt'):
            file_path_list.append(os.path.join(seed_path, f))

# sort file_path_list by AR (extract AR from filename now)
# filename format: x_relaxed_AR{AR}.txt
# We need a robust sort key.
def get_AR_value(path):
    filename = path.split('/')[-1]
    try:
        ar_str = filename.split('_AR')[1].split('.txt')[0]
        return float(ar_str)
    except:
        return 0.0

file_path_list.sort(key=get_AR_value, reverse=True)

def get_N_from_file_path(file_path):
    # Path format: .../N{N}/{SEED}/x_relaxed_AR{AR}.txt
    # N is in the 3rd to last component (parent of parent)
    parent_of_parent = file_path.split('/')[-3] # 'N20'
    return parent_of_parent.replace('N', '')

def get_AR_from_file_path(file_path):
    # filename: x_relaxed_AR{AR}.txt
    filename = file_path.split('/')[-1]
    return filename.split('_AR')[1].split('.txt')[0]

def get_random_keys_from_file_path(file_path):
    # Seed (random_keys) is the parent folder
    return file_path.split('/')[-2]


# 20251110-0333_RUN_keys919,461,568_N0200_mu0.4000_AR0300_A0.010_seed720
# 20251110-0333_RUN_keys919,461,568_N0200_mu1.0000_AR0500_A0.030_seed720
friction_list = [0.4, 1]
amplitude_list = [0.1]

# friction_list = [0.25,0.275,0.3,0.325,0.35]
# friction_list = [0.21,0.22,0.23,0.24]
# amplitude_list = [0.003, 0.005, 0.01]

# friction_list = [0.05, 0.1, 0.15, 0.18, 0.2, 0.225, 0.25, 0.3, 0.4, 1.]


# THE LIST
# friction_list = [0.01, 0.05, 0.1, 0.2, 0.4, 1]
# amplitude_list = [0.01,0.03]

# friction_list = [0.05,0.2,0.4]
# amplitude_list = [0.01]


# amplitude_list = [0.01]
# copy perturb_rod_packings.py

import yaml
# friction = 0.4

# two amplitudes
# three random keys

num_frames_stored = 2000
TIMESTEP = 0.01
seed = 720

# Toggle periodic vs non-periodic simulations here.
# When True, perturb_rod_packings will wrap rods in a periodic box
# using options["periodic"] and options["periodic_box_size"].
USE_PERIODIC = False

# Default periodic box size (only used if USE_PERIODIC is True)
PERIODIC_BOX_SIZE = [2.0, 2.0, 2.0]

# Parse arguments
import argparse

parser = argparse.ArgumentParser(description="Run simulations with optional job name.")
# default job name with timestamp
default_job_name = f"perturb_packings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
parser.add_argument("--job-name", type=str, default=default_job_name, help="Name of the job (folder name)")

args = parser.parse_args()
job_name = args.job_name

for file_path in file_path_list:
    random_keys = get_random_keys_from_file_path(file_path) # Extract dynamically
    n = get_N_from_file_path(file_path)
    
    for amplitude in amplitude_list:
        # t_u = np.sqrt( np.sqrt(2) - 1) / 2 / amplitude
        t_u = ( 2**0.5 - 1)**(0.5) / 2 / amplitude
        time_horizon = 100 * t_u
        max_steps = int(time_horizon / TIMESTEP)
        intercept = int(max_steps / num_frames_stored)

        for friction in friction_list:
            options = {
                    "file_path": file_path,
                    "mass": 1,
                    "gravity": [0, 0, 0],
                    "friction": friction,
                    "timestep": TIMESTEP,
                    "max_steps": max_steps,
                    "add_ground_plane": False,
                    "add_box_boundaries": False,
                    "save_all_data": False, # turn on only for a single run / too heavy, again...
                    "all_data_interval": 1000,
                    "realtime_visualization": False,
                    "initial_kick": True,
                    "intercept": intercept,
                    "random_seed": seed,
                    "random_amplitude": [amplitude]*6,                    
                    "periodic": USE_PERIODIC, # Periodic box controls (consumed by perturb_rod_packings.run)
                    "periodic_box_size": PERIODIC_BOX_SIZE,
                    "job_name": job_name,
                    "n_val": n,
                    "random_keys": random_keys
                }

            # export this to a yaml file
            simulations_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(simulations_dir, "options.yml"), "w") as f:
                yaml.dump(options, f)

            AR = get_AR_from_file_path(file_path)
            # Encode periodic/non-periodic choice in RUN_ID and pass as second arg to run.sh
            mode_tag = "periodic" if USE_PERIODIC else "nonperiodic"
            RUN_ID = f"keys{random_keys}_N{n}_mu{friction:.4f}_AR{AR}_A{amplitude:.3f}_seed{seed}_{mode_tag}"
            
            # sh run.sh NOTE MODE_TAG JOB_NAME N SEED
            subprocess.run(["sh", "run.sh", RUN_ID, mode_tag, job_name, str(n), str(random_keys)], cwd=simulations_dir, check=True)
        

    #         break
    #     break
    # break