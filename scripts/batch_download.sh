#!/bin/bash

# Create local destination directory
mkdir -p relax3rd_N1000_sweep

# Download all directories in a single sftp session (only one password prompt)
sftp sftp://yjung@odyssey.rc.fas.harvard.edu <<EOT
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax3rd_second_complete_run_analysis/relax3rd_N1000_sweep
lcd relax3rd_N1000_sweep
get -r 20260101-135400_355_359_829_AR300_Friction0.0_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction0.05_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction0.1_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction0.15_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction0.2_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction0.4_Kick0.1
get -r 20260101-135400_355_359_829_AR300_Friction1.0_Kick0.1
bye
EOT
