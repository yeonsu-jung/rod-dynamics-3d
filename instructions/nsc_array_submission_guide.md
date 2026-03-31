# Slurm Array Submission Guide for NSC Frictional Dynamics & Free Rod Reptation

This directory contains the instruction manual for submitting massive parametric scans natively through Slurm Job Arrays. Instead of executing sequentially bundled Slurm jobs that hold entire threads hostage for hours, these scripts dispatch up to 10,000 parallel instances into standard `#SBATCH --array=1-M` pipelines.

Everything is centralized inside `parametric_study/`.

## 1. Bulk Entangled Packings (`iter_submit_n_array.sh`) 

This wrapper executes the generalized frictional solver sweep over `parametric_study/submit_entangled_array.py`. 

### Usage & Features:
- **Dynamic Resource Sizing:** It scales CPU multithreading automatically according to the packing complexity `$N`:
  - `N < 100`: 4 Cores
  - `100 <= N <= 200`: 8 Cores
  - `N = 500`: 16 Cores
  - `N = 1000`: 32 Cores
- **Massive Array Dispatch:** Spawns an automated `Master_Sbatch.sh` list for a single `$N` configuration. 

**Execution:**
You do not need massive CLI arguments. Simply execute:
```bash
bash parametric_study/iter_submit_n_array.sh
```
*Note: Make sure to modify the `LIMIT` toggle in the script to scale from testing limits (`LIMIT=5`) up to your full massive deployments (`LIMIT=0`).*


## 2. Free Rod Reptation (`iter_submit_free_rod_array.sh`)

This script extracts specific rod conditions from `extreme_rods_summary.csv` (`MinFSA`, `MaxFTA`, etc.) and runs them through `submit_free_rod_array.py`.

### Usage & Features:
- **Single Rod Path Processing:** Maps `--fix-every-except` universally across the maze so that the single free rod reptation can be monitored cleanly via the `--test-rod-endpoints` endpoint logging structure. 
- **Array Safe Spacing:** The wrapper parses configs natively and builds sequential arrays *strictly per N limit* ensuring you do not exceed max Slurm array caps, stopping reliably before any config exceeds `N=1000`. 
- **Fixed Cores:** Retains exactly 8 CPU cores perfectly sufficient for the static maze broadphase geometry.

**Execution Parameters:**
We universally pipe in the `$N` values, `$ALPHA` geometries, `frictions`, and `--nsc` initialization targets cleanly here:
```bash
bash parametric_study/iter_submit_free_rod_array.sh \
  --n-list "10,15,20,30,50,100,200,500,1000" \
  --alpha-list "10,25,50,100,150,200,300,500,1000" \
  --ids-per-n 1 \
  --job-name free_rod_array_run \
  --frictions "0.0,0.05,0.1,0.15,0.2,0.4,1.0" \
  --frames 200000 \
  --endpoint-stride 1000 \
  --endpoint-max 0 \
  -- --nsc --sigma-v 0.1 --nsc-iters 40 --nsc-beta 0.2 --nsc-pos-iters 5 --nsc-pos-psor 50
```

### Computing the Sliding Path Length Offset
Using `--test-rod-endpoints` outputs lightweight CSV dumps `(step, x1,y1,z1, x2,y2,z2)`. Because of this structure, tracking sliding offset is completely frictionless during post processing:
1. Tangent Orientation: `u_t = normalize(P2_t - P1_t)`
2. Centroid Vector: `C_t = (P1_t + P2_t) / 2`
3. Sliding Path Update per Frame: `|dot( C_t - C_{t-1}, u_{t-1} )|` 

---
**Core Dependencies:**
* `parametric_study/submit_entangled_array.py` 
* `parametric_study/submit_free_rod_array.py`
These generate the underlying `array_commands.txt` file and proxy directly into single-instance `Sbatch` nodes automatically!
