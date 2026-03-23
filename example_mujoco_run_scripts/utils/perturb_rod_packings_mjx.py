"""Perturb rod packings with GPU-accelerated MuJoCo MJX.

This is the GPU counterpart of perturb_rod_packings.py. It uses MuJoCo MJX
(JAX-based) to run mj_step on GPU, providing significant speedups for
contact-rich simulations (N >= 200).

Key differences from the CPU version:
  - Uses mjx.step() instead of mujoco.mj_step()
  - Integrator must be 'euler' or 'implicit' (MJX doesn't support RK4)
  - Data lives on GPU; pulled to CPU only at recording intervals
  - Requires JAX with CUDA support

Usage:
  python perturb_rod_packings_mjx.py  # reads options.yml
"""

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

import numpy as np
import csv
import time
import copy
import os
import datetime

from packing_initialization import read_and_convert_to_co_format, read_from_file


def get_geom_name(model, geom_id):
    name_offset = model.name_geomadr[geom_id]
    return model.names[name_offset:].split(b'\0', 1)[0].decode('utf-8')


def create_mujoco_model_from_file(file_path, options=None):
    """Build an MJCF XML model from rod packing data.
    
    This is identical to the CPU version EXCEPT:
      - integrator is set to 'implicit' instead of 'RK4' (MJX requirement)
    """
    if options is None:
        options = {
            "add_ground_plane": False,
            "add_box_boundaries": False,
        }

    g = options["gravity"]
    m = options["mass"]
    friction = options["friction"]

    # --- Parse aspect ratio (AR) from path ---
    parent_name = file_path.split('/')[-2]
    ar = None
    if '-AR' in parent_name:
        try:
            ar_str = parent_name.split('-AR')[1].split('-')[0]
            ar = float(ar_str)
        except (IndexError, ValueError):
            ar = None

    if ar is None and '_AR' in parent_name:
        try:
            after = parent_name.split('_AR')[1]
            digits = ''.join(ch for ch in after if ch.isdigit())
            if digits:
                ar = float(digits)
        except Exception:
            ar = None

    # Fallback: parse AR from filename
    if ar is None:
        filename = file_path.split('/')[-1]
        if 'AR' in filename:
            try:
                after_ar = filename.split('AR')[1]
                ar_str = after_ar.split('.txt')[0].split('_')[0]
                ar = float(ar_str)
            except (IndexError, ValueError):
                pass

    if ar is None:
        raise ValueError(
            f"Could not parse AR from parent directory name '{parent_name}' "
            f"or filename '{file_path.split('/')[-1]}'"
        )

    # Try to parse rod_radius from file header
    rod_radius_from_file = None
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# rod_radius ='):
                rod_radius_from_file = float(first_line.split('=')[1].strip())
    except Exception as e:
        print(f"Could not parse rod_radius from file header: {e}")

    rod_radius = options.get("rod_radius", None)
    rod_length = options.get("rod_length", 1.0)

    if rod_radius is None:
        if rod_radius_from_file is not None:
            radius = rod_radius_from_file
        else:
            radius = 1 / ar / 2
    else:
        radius = float(rod_radius)

    two_points_data = read_from_file(file_path)
    co_data = read_and_convert_to_co_format(file_path)

    # MJX-compatible integrator (only 'Euler' and 'implicitfast' supported)
    # MuJoCo MJCF uses: 'Euler', 'RK4', 'implicit', 'implicitfast'
    integrator = options.get("mjx_integrator", "Euler")

    xml = f"""
    <mujoco model="particles_in_box">
        <option timestep="{options['timestep']}" gravity="{g[0]} {g[1]} {g[2]}" integrator="{integrator}">
        </option>

        <visual>
            <global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/>
            <map znear="0.01" zfar="50"/>
        </visual>
        <default>
            <geom friction="{friction} 0. 0."/> <!-- Apply default friction settings -->
        </default>
        <worldbody>
            <!-- Light -->
            <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    """

    global_centroid = np.mean(
        (two_points_data[:, :3] + two_points_data[:, 3:]) / 2, axis=0
    )

    print(f"[MJX] Bounding box max: {np.max(two_points_data[:,:3] - global_centroid, axis=0)}")
    print(f"[MJX] Bounding box min: {np.min(two_points_data[:,:3] - global_centroid, axis=0)}")

    if options.get("add_ground_plane", False):
        box_bottom = -0.65
        xml += f"""
        <geom name="ground" type="plane" pos="0 0 {box_bottom}" size="5 5 0.1" rgba="0.5 0.5 0.5 0.5"/>
        """

    if options.get("add_box_boundaries", False):
        box_halfsize = 0.75
        box_bottom = -0.65
        xml += f"""
        <geom name="wall_x_pos" type="box" pos=" {box_halfsize} 0   0"  size="0.05 {box_halfsize} {box_halfsize}" rgba="1 0 0 0.1"/>
        <geom name="wall_x_neg" type="box" pos="-{box_halfsize} 0   0"  size="0.05 {box_halfsize} {box_halfsize}"   rgba="0 1 0 0.1"/>
        <geom name="wall_y_pos" type="box" pos=" 0   {box_halfsize} 0"  size="{box_halfsize}  0.05 {box_halfsize}"   rgba="0 0 1 0.1"/>
        <geom name="wall_y_neg" type="box" pos=" 0  -{box_halfsize} 0"  size="{box_halfsize}  0.05 {box_halfsize}"   rgba="1 1 0 0.1"/>
        <geom name="wall_z_top" type="box" pos="0 0 {2*box_halfsize + box_bottom}" size="5 5 0.05" rgba="1 0 1 0.1"/>
        """

    num_particles = co_data.shape[0]

    for i in range(num_particles):
        x, y, z = co_data[i, 0], co_data[i, 1], co_data[i, 2]
        x -= global_centroid[0]
        y -= global_centroid[1]
        z -= global_centroid[2]
        u, v, w = co_data[i, 3], co_data[i, 4], co_data[i, 5]
        xml += f"""
        <body name="particle{i}" pos="{x} {y} {z}">
            <geom name="particle{i}" type="cylinder" size="{radius} {rod_length/2.0}" mass="{m}" rgba="0.8 0.8 0.8 1" zaxis="{u} {v} {w}"/>
            <joint type="free"/>
        </body>
        """
    xml += "</worldbody></mujoco>"
    return xml


def get_rod_coordinates_from_data(data):
    """Extract rod centroids and orientations from MuJoCo data."""
    centroids = data.geom_xpos
    orientations = data.geom_xmat
    return centroids, orientations


def run(file_path, output_path, options):
    """Run GPU-accelerated simulation using MJX.
    
    This is the core simulation loop. It:
      1. Builds the MuJoCo model from the rod packing file
      2. Moves model and data to GPU via MJX
      3. Runs jit-compiled mjx.step() on GPU
      4. Periodically pulls data back to CPU for recording
    """

    # --- Build model ---
    xml_model = create_mujoco_model_from_file(file_path, options=options)
    with open("rod_model.xml", "w") as f:
        f.write(xml_model)

    model = mujoco.MjModel.from_xml_string(xml_model)
    data = mujoco.MjData(model)

    print(f"[MJX] Model: {model.nbody} bodies, {model.ngeom} geoms, {model.nq} qpos, {model.nv} qvel")
    print(f"[MJX] JAX devices: {jax.devices()}")

    # --- Initial kick (on CPU, before moving to GPU) ---
    random_seed = options["random_seed"]
    random_amplitude = options["random_amplitude"]
    np.random.seed(random_seed)

    if options.get("initial_kick", False):
        for i in range(model.nv // 6):
            idx = 6 * i
            t_sigmas = np.array(random_amplitude[:3], dtype=float)
            v = np.random.normal(loc=0.0, scale=t_sigmas)

            rot_sigmas = np.array(random_amplitude[3:6], dtype=float)
            sigma_rot = float(np.mean(rot_sigmas))
            dir_vec = np.random.normal(size=3)
            norm = np.linalg.norm(dir_vec)
            if norm == 0.0:
                dir_vec = np.array([1.0, 0.0, 0.0])
            else:
                dir_vec = dir_vec / norm
            magnitude = np.random.normal(loc=0.0, scale=sigma_rot)
            w = magnitude * dir_vec

            data.qvel[idx + 0] = v[0]
            data.qvel[idx + 1] = v[1]
            data.qvel[idx + 2] = v[2]
            data.qvel[idx + 3] = w[0]
            data.qvel[idx + 4] = w[1]
            data.qvel[idx + 5] = w[2]

    # --- Move to GPU ---
    print("[MJX] Moving model and data to GPU...")
    t_gpu_start = time.time()
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    print(f"[MJX] GPU transfer took {time.time() - t_gpu_start:.3f}s")

    # --- JIT compile the step function ---
    print("[MJX] JIT-compiling mjx.step (first call will be slow)...")
    jit_step = jax.jit(mjx.step)

    # Warm-up: trigger JIT compilation
    t_jit_start = time.time()
    mjx_data = jit_step(mjx_model, mjx_data)
    mjx_data.qpos.block_until_ready()  # Force synchronization
    print(f"[MJX] JIT compilation took {time.time() - t_jit_start:.3f}s")

    # --- Output file setup ---
    centroids_file = f'{output_path}/centroids_over_time.txt'
    orientations_file = f'{output_path}/orientations_over_time.txt'
    time_points_file = f'{output_path}/time_points.txt'
    xipos_file = f'{output_path}/xipos_over_time.txt'
    xmat_file = f'{output_path}/xmat_over_time.txt'
    qvel_file = f'{output_path}/qvel_over_time.txt'
    energy_file = f'{output_path}/energy_over_time.txt'
    contact_csv = f"{output_path}/rod_contact_info.csv"

    csv_columns = [
        "step", "geom1", "geom2",
        "pos", "force", "torque", "frame", "dist", "includemargin",
        "friction", "mu", "solref", "solimp"
    ]
    with open(contact_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # --- Main simulation loop ---
    max_steps = options["max_steps"]
    intercept = options["intercept"]

    print(f"[MJX] Starting simulation: {max_steps} steps, recording every {intercept} steps")
    t_sim_start = time.time()
    steps_since_last_report = 0
    t_last_report = t_sim_start

    try:
        for step in range(1, max_steps + 2):
            # GPU step
            mjx_data = jit_step(mjx_model, mjx_data)

            # Record data at intervals
            if (step - 1) % intercept == 0:
                # Pull data back to CPU for recording
                cpu_data = mjx.get_data(model, mjx_data)

                # Timing report
                t_now = time.time()
                elapsed = t_now - t_sim_start
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                interval_steps = steps_since_last_report
                interval_time = t_now - t_last_report
                interval_sps = interval_steps / interval_time if interval_time > 0 else 0

                print(
                    f"[MJX] Step {step}/{max_steps} | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Avg: {steps_per_sec:.0f} steps/s | "
                    f"Interval: {interval_sps:.0f} steps/s"
                )
                steps_since_last_report = 0
                t_last_report = t_now

                # Extract centroids and orientations
                centroids = np.array(cpu_data.geom_xpos)
                orientations = np.array(cpu_data.geom_xmat)

                with open(centroids_file, 'a') as f:
                    np.savetxt(f, centroids.flatten()[None], fmt='%.8e', delimiter=',')
                with open(orientations_file, 'a') as f:
                    np.savetxt(f, orientations.flatten()[None], fmt='%.8e', delimiter=',')
                with open(time_points_file, 'a') as f:
                    np.savetxt(f, [float(cpu_data.time)], fmt='%.8e')
                with open(xipos_file, 'a') as f:
                    np.savetxt(f, np.array(cpu_data.xipos).flatten()[None], fmt='%.8e', delimiter=',')
                with open(xmat_file, 'a') as f:
                    np.savetxt(f, np.array(cpu_data.xmat).flatten()[None], fmt='%.8e', delimiter=',')
                with open(qvel_file, 'a') as f:
                    np.savetxt(f, np.array(cpu_data.qvel).flatten()[None], fmt='%.8e', delimiter=',')
                with open(energy_file, 'a') as f:
                    # Energy flag disabled for MJX compat; write zeros as placeholder
                    try:
                        energy = np.array(cpu_data.energy)
                    except Exception:
                        energy = np.array([0.0, 0.0])
                    np.savetxt(f, [energy], fmt='%.8e', delimiter=',')

                # Contact info
                contact_rows = []
                if cpu_data.ncon > 0:
                    for ci in range(cpu_data.ncon):
                        contact = cpu_data.contact[ci]
                        normal = contact.frame[:3]
                        tangent1 = contact.frame[3:6]
                        tangent2 = contact.frame[6:9]

                        c_array = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(model, cpu_data, ci, c_array)

                        force_w = c_array[0]*normal + c_array[1]*tangent1 + c_array[2]*tangent2
                        torque_w = c_array[3]*normal + c_array[4]*tangent1 + c_array[5]*tangent2

                        contact_rows.append({
                            'step': step,
                            'geom1': contact.geom1,
                            'geom2': contact.geom2,
                            'dist': contact.dist,
                            'mu': contact.mu,
                            'includemargin': contact.includemargin,
                            'pos': ','.join(map(str, contact.pos)),
                            'frame': ','.join(map(str, contact.frame.flatten())),
                            'force': ','.join(map(str, force_w)),
                            'torque': ','.join(map(str, torque_w)),
                            'friction': ','.join(map(str, contact.friction)),
                            'solref': ','.join(map(str, contact.solref)),
                            'solimp': ','.join(map(str, contact.solimp)),
                        })
                else:
                    contact_rows.append({
                        'step': step, 'geom1': -1, 'geom2': -1,
                        'dist': -1, 'mu': -1, 'includemargin': -1,
                        'pos': -1, 'force': -1, 'torque': -1,
                        'frame': -1, 'friction': -1, 'solref': -1, 'solimp': -1
                    })

                with open(contact_csv, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_columns)
                    writer.writerows(contact_rows)
            else:
                steps_since_last_report += 1

        t_total = time.time() - t_sim_start
        print(f"[MJX] Simulation completed: {max_steps} steps in {t_total:.1f}s "
              f"({max_steps/t_total:.0f} steps/s)")

    except Exception as e:
        print(f"[MJX] Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save final state (pull from GPU)
        cpu_data = mjx.get_data(model, mjx_data)

        import pickle
        with open(f'{output_path}/data.pickle', 'wb') as f:
            pickle.dump(cpu_data, f)
        with open(f'{output_path}/model.pickle', 'wb') as f:
            pickle.dump(model, f)

        return output_path


if __name__ == "__main__":
    import yaml
    import shutil
    import re
    from pathlib import Path

    with open('options.yml', 'r') as f:
        options = yaml.safe_load(f)
    print(f"[MJX] Options: {options}")

    file_path = options["file_path"]
    file_id = Path(file_path).parent.name

    match = re.search(r'\d+,\d+,\d+', file_path)
    random_keys = match.group(0) if match else 'unknown'

    if os.name == 'posix' and 'darwin' not in os.uname().sysname.lower():
        storage_root = '/n/holylabs/LABS/mahadevan_lab/Users/yjung/maximal-entanglement/'
        current_folder = os.getcwd()
        job_name = options.get('job_name', 'default_job')
        n_val = options.get('n_val', 'unknown_N')
        random_keys_opt = options.get('random_keys', 'unknown_keys')
        leaf_folder_name = current_folder.split('/')[-1]
        dest_dir = os.path.join(
            storage_root, job_name, f"N{n_val}", str(random_keys_opt), leaf_folder_name
        )
        if not os.path.exists(dest_dir):
            shutil.copytree(current_folder, dest_dir)
        else:
            print(f"Destination {dest_dir} already exists.")
        output_path = f'{dest_dir}/outputs/{random_keys}/{file_id}_'
    else:
        output_path = f'outputs/{random_keys}/{file_id}_'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f'[MJX] Output directory: {output_path}')

    shutil.copyfile(file_path, f'{output_path}/x_relaxed.txt')

    print(f'[MJX] MuJoCo version: {mujoco.__version__}')
    print(f'[MJX] JAX version: {jax.__version__}')
    print(f'[MJX] JAX devices: {jax.devices()}')

    output_path = run(file_path=file_path, output_path=output_path, options=options)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/output.txt', 'w') as f:
        f.write(f'MuJoCo version: {mujoco.__version__}\n')
        f.write(f'JAX version: {jax.__version__}\n')
        f.write(f'JAX devices: {jax.devices()}\n')
        f.write(f'Backend: MJX (GPU)\n')

    with open(f'{output_path}/options.txt', 'w') as f:
        f.write(str(options))

    shutil.copyfile(__file__, f'{output_path}/perturb_rod_packings_mjx.py')

    with open(f'{output_path}/mujoco_version.txt', 'w') as f:
        f.write(mujoco.__version__)
