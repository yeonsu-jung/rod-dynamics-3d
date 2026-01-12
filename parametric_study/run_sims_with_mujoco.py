"""Perturb rod packings and record contact/trajectory data.

Now optionally supports periodic wrapping of free-root rod bodies,
inspired by `periodic_cylinders.py`.

TODO: start with an initial velocity and move forward a single step
to see how "contact" network is formed.
"""

import mujoco

import numpy as np
import csv
# from packing_initialization import read_and_convert_to_co_format,read_from_file
import time
import copy
import traceback


def read_and_convert_to_co_format(file_path):
    two_points_data = read_from_file(file_path)

    # if two_points_data is a matrix
    if len(two_points_data.shape) == 2:
        n = two_points_data.shape[0]
    else:
        n = 1
        two_points_data = two_points_data.reshape(1,6)

    co_data = np.zeros((n,6))
    for i in range(n):
        r1 = two_points_data[i,0:3]
        r2 = two_points_data[i,3:6]
        centroid = (r1+r2)/2
        orientation = (r2-r1)/np.linalg.norm(r2-r1)
        co_data[i,0:3] = centroid
        co_data[i,3:] = orientation
    return co_data

def read_from_file(file_path):
    dta = np.loadtxt(file_path, delimiter=' ', comments='#')
    return dta

def get_geom_name(model,geom_id):
    name_offset = model.name_geomadr[geom_id]
    return model.names[name_offset:].split(b'\0', 1)[0].decode('utf-8')


# ---------------- Periodic wrapping utilities ---------------- #

def _wrap_coord(x, L):
    """Wrap scalar x into [0, L). Robust to negative values."""
    return x - np.floor(x / L) * L


def _wrap_body_freejoint_periodic(model, data, body_id, box_size):
    """Wrap a free-root body into a periodic box by modifying its qpos.

    Assumes body has a free joint at its root.
    """
    jnt_id = model.body_jntadr[body_id]
    if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            f"Body id {body_id} does not have a free joint (periodic wrapper assumes free bodies)"
        )

    adr = model.jnt_qposadr[jnt_id]  # index into qpos for this joint
    # qpos[adr:adr+3] is translation; qpos[adr+3:adr+7] is quaternion
    pos = data.qpos[adr:adr+3].copy()

    for i in range(3):
        pos[i] = _wrap_coord(pos[i], box_size[i])

    data.qpos[adr:adr+3] = pos


def _step_periodic(model, data, box_size, body_ids):
    """Single simulation step with periodic wrapping."""
    mujoco.mj_step(model, data)

    # wrap positions for all selected bodies
    for bid in body_ids:
        _wrap_body_freejoint_periodic(model, data, bid, box_size)

    # recompute derived quantities after modifying qpos
    mujoco.mj_forward(model, data)

def save_contact_info(model, data, step):
    """Legacy helper to dump detailed contact/state info (currently unused).

    Kept for debugging reference; not wired into the main run loop.
    """
    contact_info = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = get_geom_name(model, contact.geom1)
        geom2_name = get_geom_name(model, contact.geom2)

        position = contact.pos
        normal = contact.frame[:3]
        tangent1 = contact.frame[3:6]
        tangent2 = contact.frame[6:9]

        c_array = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, c_array)

        force_in_world_frame = (
            c_array[0] * normal +
            c_array[1] * tangent1 +
            c_array[2] * tangent2
        )
        torque_in_world_frame = (
            c_array[3] * normal +
            c_array[4] * tangent1 +
            c_array[5] * tangent2
        )

        contact_info.append({
            "step": step,
            "geom1": geom1_name,
            "geom2": geom2_name,
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
            "force_x": force_in_world_frame[0],
            "force_y": force_in_world_frame[1],
            "force_z": force_in_world_frame[2],
            "torque_x": torque_in_world_frame[0],
            "torque_y": torque_in_world_frame[1],
            "torque_z": torque_in_world_frame[2],
        })

    csv_file = "rod_contact_info.csv"
    csv_columns = [
        "step", "geom1", "geom2", "position_x", "position_y", "position_z",
        "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z",
    ]
    try:
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(contact_info)
    except IOError:
        # Non-fatal: just report and continue.
        print("I/O error when saving contact information.")

def create_mujoco_model_from_file(file_path,options=None):
    if options == None:
        options = {
            "add_ground_plane": False,
            "add_box_boundaries": False
        }

    g = options["gravity"]
    m = options["mass"]
    friction = options["friction"]


    
    # Priority 1: AR from options (passed via CLI)
    ar = options.get("ar", None)

    # Priority 2: Parse from file path
    if ar is None:
        try:
            parts = file_path.split('/')
            if len(parts) > 1:
                parent_name = parts[-2]
            else:
                parent_name = ""

            # Robustly parse aspect ratio (AR) from parent folder name.
            if '-AR' in parent_name:
                try:
                    ar_str = parent_name.split('-AR')[1].split('-')[0]
                    ar = float(ar_str)
                except (IndexError, ValueError):
                    pass

            if ar is None and '_AR' in parent_name:
                try:
                    after = parent_name.split('_AR')[1]
                    digits = ''.join(ch for ch in after if ch.isdigit())
                    if digits:
                        ar = float(digits)
                except Exception:
                    pass
        except Exception:
            pass

    if ar is None and '_AR' in parent_name:
        try:
            after = parent_name.split('_AR')[1]
            digits = ''.join(ch for ch in after if ch.isdigit())
            if digits:
                ar = float(digits)
        except Exception:
            ar = None

    # Fallback: try parsing AR from the filename itself (e.g., x_relaxed_AR1000.txt)
    if ar is None:
        filename = file_path.split('/')[-1]
        if 'AR' in filename:
             try:
                # e.g., "x_relaxed_AR1000.txt" -> "1000"
                # Split by 'AR', take the part after.
                # Then take the leading digits or digits before '.txt'
                after_ar = filename.split('AR')[1]
                # If there's extensions or other suffixes, clean it up
                # Simple heuristic: take digits until non-digit (except maybe dot?)
                # actually, split by '.txt' usually works
                ar_str = after_ar.split('.txt')[0]
                # remove any other underscores if present?
                ar_str = ar_str.split('_')[0]
                ar = float(ar_str)
             except (IndexError, ValueError):
                pass

    if ar is None:
        raise ValueError(f"Could not parse AR from parent directory name '{parent_name}' or filename '{file_path.split('/')[-1]}'")

    # Try to parse rod_radius from the first line of the file
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# rod_radius ='):
                rod_radius_from_file = float(first_line.split('=')[1].strip())
    except Exception as e:
        print(f"Could not parse rod_radius from file header: {e}")
        rod_radius_from_file = None

    # Optionally allow overriding radius/length from options; otherwise, use AR-based radius.
    rod_radius = options.get("rod_radius", None) # options takes precedence? Or file? 
    # Usually explicit options override file, but if options is None, check file.
    
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

    xml = f"""
    <mujoco model="particles_in_box">
        <option timestep="{options["timestep"]}" gravity="{g[0]} {g[1]} {g[2]}" integrator="RK4">
        <flag energy="enable"/>
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
            
            <!-- Camera -->
            <!-- <camera name="fixed" pos="100 100 5" euler="0 0 0"/> -->
    """
    
    global_centroid = np.mean((two_points_data[:,:3] + two_points_data[:,3:])/2,axis=0)

    print(np.max(two_points_data[:,:3] - global_centroid,axis=0))
    print(np.min(two_points_data[:,:3] - global_centroid,axis=0))

    print(np.max(two_points_data[:,3:] - global_centroid,axis=0))
    print(np.min(two_points_data[:,3:] - global_centroid,axis=0))

    # np.min(two_points_data[:,:3] - global_centroid,axis=0)
    # np.min(two_points_data[:,3:] - global_centroid,axis=0)

    box_bottom = -0.65
    box_halfsize = 0.75

    if options["add_ground_plane"]:
        xml += f"""
        <!-- Ground plane -->
        <geom name="ground" type="plane" pos="0 0 {box_bottom}" size="5 5 0.1" rgba="0.5 0.5 0.5 0.5"/>
        """
    if options["add_box_boundaries"]:        
        xml += f"""
        <!-- Box boundaries -->
        <geom name="wall_x_pos" type="box" pos=" {box_halfsize} 0   0"  size="0.05 {1*box_halfsize} {1*box_halfsize}" rgba="1 0 0 0.1"/>
        <geom name="wall_x_neg" type="box" pos="-{box_halfsize} 0   0"  size="0.05 {1*box_halfsize} {1*box_halfsize}"   rgba="0 1 0 0.1"/>
        <geom name="wall_y_pos" type="box" pos=" 0   {box_halfsize} 0"  size="{1*box_halfsize}  0.05 {1*box_halfsize}"   rgba="0 0 1 0.1"/>
        <geom name="wall_y_neg" type="box" pos=" 0  -{box_halfsize} 0"  size="{1*box_halfsize}  0.05 {1*box_halfsize}"   rgba="1 1 0 0.1"/>

        <geom name="wall_z_top" type="box" pos="0 0 {2*box_halfsize + box_bottom}" size="5 5 0.05" rgba="1 0 1 0.1"/>
        """

    num_particles = co_data.shape[0]

    # Instantiate each rod as a cylinder with configurable radius/length.
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
    centroids = data.geom_xpos
    orientations = data.geom_xmat

    return centroids, orientations


def run(file_path, output_path, options):
    import datetime
    import os

    # output_path = output_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'    
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # copy the file to the output directory
    # import shutil
    # shutil.copyfile(file_path, output_path + 'x_relaxed.txt')

    # Create the MuJoCo model
    xml_model = create_mujoco_model_from_file(file_path,options=options)

    # export xml
    with open("rod_model.xml", "w") as f:
        f.write(xml_model)

    model = mujoco.MjModel.from_xml_string(xml_model)
    data = mujoco.MjData(model)

    # Periodic box configuration (optional)
    use_periodic = bool(options.get("periodic", False))
    periodic_box_size = np.array(options.get("periodic_box_size", [2.0, 2.0, 2.0]), dtype=float)

    # Precompute body ids for periodic wrapping if enabled
    periodic_body_ids = None
    if use_periodic:
        # Here we wrap all free-joint bodies corresponding to rods.
        # They are named "particle{i}" in create_mujoco_model_from_file.
        periodic_body_ids = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                continue
            if name.startswith("particle"):
                periodic_body_ids.append(i)

        print(f"[periodic] Enabled periodic wrapping for {len(periodic_body_ids)} bodies")

    # random keys
    random_seed = options["random_seed"]
    random_amplitude = options["random_amplitude"] # must be an array of size 6 (qvel dim)


    np.random.seed(random_seed)
    if options["initial_kick"]:
        # Assign initial velocities from stochastic distributions:
        # - Translational (vx, vy, vz): Gaussian N(0, sigma^2) with axis-wise std devs random_amplitude[0:3]
        # - Rotational (wx, wy, wz): Magnitude ~ N(0, sigma_rot^2), direction ~ uniform on S2
        #   where sigma_rot = mean(random_amplitude[3:6]) for isotropy.
        for i in range(model.nv//6):
            idx = 6*i
            # Translational 3-vector
            t_sigmas = np.array(random_amplitude[:3], dtype=float)
            v = np.random.normal(loc=0.0, scale=t_sigmas)

            # Rotational 3-vector
            rot_sigmas = np.array(random_amplitude[3:6], dtype=float)
            sigma_rot = float(np.mean(rot_sigmas))
            # sample a random unit direction on S2
            dir_vec = np.random.normal(size=3)
            norm = np.linalg.norm(dir_vec)
            # extremely unlikely, but guard against zero norm
            if norm == 0.0:
                dir_vec = np.array([1.0, 0.0, 0.0])
            else:
                dir_vec = dir_vec / norm
            magnitude = np.random.normal(loc=0.0, scale=sigma_rot)
            w = magnitude * dir_vec

            # Write into qvel (free joint per body: 3 translational, 3 angular)
            data.qvel[idx+0] = v[0]
            data.qvel[idx+1] = v[1]
            data.qvel[idx+2] = v[2]
            data.qvel[idx+3] = w[0]
            data.qvel[idx+4] = w[1]
            data.qvel[idx+5] = w[2]

    # Helper function to get geom names
    def get_geom_name(geom_id):
        name_offset = model.name_geomadr[geom_id]
        return model.names[name_offset:].split(b'\0', 1)[0].decode('utf-8')

    data_all = []
    data_all.append(copy.deepcopy(data))

    centroids_over_time = []
    orientations_over_time = []

    time_points = []
    xipos_over_time = []
    xmat_over_time = []
    contact_over_time = []
    qvel_over_time = []
    energy_over_time = []

    time_points_txt_file = (f'{output_path}/time_points.txt')
    centroids_over_time_txt_file = (f'{output_path}/centroids_over_time.txt')
    orientations_over_time_txt_file = (f'{output_path}/orientations_over_time.txt')
    xipos_over_time_txt_file = (f'{output_path}/xipos_over_time.txt')
    xmat_over_time_txt_file = (f'{output_path}/xmat_over_time.txt')
    qvel_over_time_txt_file = (f'{output_path}/qvel_over_time.txt')
    energy_over_time_txt_file = (f'{output_path}/energy_over_time.txt')
    contact_over_time_txt_file = (f'{output_path}/contact_over_time.txt')

    file_exists = os.path.isfile(centroids_over_time_txt_file)
    csv_columns = [
                    "step", "geom1", "geom2",
                    "pos", "force", "torque", "frame", "dist", "includemargin",
                    "friction", "mu", "solref", "solimp"
                ]
    
    csv_file = f"{output_path}/rod_contact_info.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        if not file_exists:
            writer.writeheader()  # Only write header if file doesn't exist

    try:
        for step in range(options["max_steps"] + 2):
            
            # save first
            if options["save_all_data"] and step % options["all_data_interval"] == 0:
                data_all.append(copy.deepcopy(data))
                
            if (step-1) % options["intercept"] == 0:
                # data.qvel[5*i+0] += np.random.uniform(-0.1, 0.1)
                # data.qvel[5*i+1] += np.random.uniform(-0.1, 0.1)
                # data.qvel[5*i+2] += np.random.uniform(-0.1, 0.1)
                # data.qvel[5*i+3] += np.random.uniform(-0.1, 0.1)
                # data.qvel[5*i+4] += np.random.uniform(-0.1, 0.1)
                # data.qvel[5*i+5] += np.random.uniform(-0.1, 0.1) 
                print(f"Step {step}/{options['max_steps']}")
                
                centroids, orientations = get_rod_coordinates_from_data(data)
                centroids_over_time.append(centroids.flatten())
                orientations_over_time.append(orientations.flatten())

                # time_points.append(data.time)
                # xipos_over_time.append(data.xipos.flatten())
                # xmat_over_time.append(data.xmat.flatten())
                # qvel_over_time.append(data.qvel.flatten())
                # energy_over_time.append(data.energy.copy())

                # Save contact information to file
                # with open(centroids_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{centroids.flatten()}\n")
                # with open(orientations_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{orientations.flatten()}\n")
                # with open(time_points_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{data.time}\n")
                # with open(xipos_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{data.xipos.flatten()}\n")
                # with open(xmat_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{data.xmat.flatten()}\n")
                # with open(qvel_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{data.qvel.flatten()}\n")
                # with open(energy_over_time_txt_file, 'a') as contact_file:
                #     contact_file.write(f"{data.energy.copy()}\n")
                # Save endpoints for crossing analysis
                # rod_length = options.get("rod_length", 1.0)
                rod_length = 1.0 # Default
                
                # centroids: (N, 3), orientations: (N, 4) [w, x, y, z]
                # Direction vector d = R * [0, 0, 1]
                # For quaternion [w, x, y, z], rotated vector [0,0,1] is:
                # [ 2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2) ]
                # Ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
                
                endpoints_csv_file = f'{output_path}/endpoints.csv'
                
                # Compute endpoints in loop
                with open(endpoints_csv_file, 'a') as f:
                    for i in range(len(centroids)):
                        c = centroids[i]
                        q = orientations[i] # w, x, y, z
                        
                        w, x, y, z = q[0], q[1], q[2], q[3]
                        
                        # Direction vector (local z-axis rotated)
                        dx = 2.0 * (x*z + w*y)
                        dy = 2.0 * (y*z - w*x)
                        dz = 1.0 - 2.0 * (x*x + y*y)
                        
                        # Half length
                        h = rod_length / 2.0
                        
                        p1x, p1y, p1z = c[0] - h*dx, c[1] - h*dy, c[2] - h*dz
                        p2x, p2y, p2z = c[0] + h*dx, c[1] + h*dy, c[2] + h*dz
                        
                        # Format: x1, y1, z1, x2, y2, z2
                        f.write(f"{p1x:.6f},{p1y:.6f},{p1z:.6f},{p2x:.6f},{p2y:.6f},{p2z:.6f}\n")

                with open(centroids_over_time_txt_file, 'a') as f:
                    np.savetxt(f, centroids.flatten()[None], fmt='%.8e', delimiter=',')

                with open(orientations_over_time_txt_file, 'a') as f:
                    np.savetxt(f, orientations.flatten()[None], fmt='%.8e', delimiter=',')

                with open(time_points_txt_file, 'a') as f:
                    np.savetxt(f, [data.time], fmt='%.8e')

                with open(xipos_over_time_txt_file, 'a') as f:
                    np.savetxt(f, data.xipos.flatten()[None], fmt='%.8e', delimiter=',')

                with open(xmat_over_time_txt_file, 'a') as f:
                    np.savetxt(f, data.xmat.flatten()[None], fmt='%.8e', delimiter=',')

                with open(qvel_over_time_txt_file, 'a') as f:
                    np.savetxt(f, data.qvel.flatten()[None], fmt='%.8e', delimiter=',')

                with open(energy_over_time_txt_file, 'a') as f:
                    np.savetxt(f, [data.energy.copy()], fmt='%.8e', delimiter=',')

                # Save contact information to file

                

                contact_list_serializable = []
                if data.ncon > 0:
                    contact_i = 0
                    for contact_instance in data.contact:
                        # contact_info = {
                        #     'pos': contact_instance.pos.copy(),
                        #     'frame': contact_instance.frame.copy(),
                        #     'geom1': contact_instance.geom1,
                        #     'geom2': contact_instance.geom2,
                        #     'dist': contact_instance.dist,
                        #     'includemargin': contact_instance.includemargin,
                        #     'friction': contact_instance.friction.copy(),
                        #     'mu': contact_instance.mu,
                        #     'solref': contact_instance.solref.copy(),
                        #     'solimp': contact_instance.solimp.copy(),
                        #     # Add more fields if you need!
                        # }

                        
                        # tangent1 = contact_instance.frame[3:6]
                        # tangent2 = contact_instance.frame[6:9]

                        # c_array = np.zeros(6, dtype=np.float64)
                        # mujoco.mj_contactForce(model, data, i, c_array)

                        # force_in_world_frame = c_array[0]*normal + c_array[1]*tangent1 + c_array[2]*tangent2
                        # torque_in_world_frame = c_array[3]*normal + c_array[4]*tangent1 + c_array[5]*tangent2

                        



                        normal = contact_instance.frame[:3]
                        tangent1 = contact_instance.frame[3:6]
                        tangent2 = contact_instance.frame[6:9]

                        c_array = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(model, data, contact_i, c_array)
                        contact_i += 1

                        force_in_world_frame = c_array[0]*normal + c_array[1]*tangent1 + c_array[2]*tangent2
                        torque_in_world_frame = c_array[3]*normal + c_array[4]*tangent1 + c_array[5]*tangent2

                        flat_contact_info = {
                            'step': step,
                            'geom1': contact_instance.geom1,
                            'geom2': contact_instance.geom2,
                            'dist': contact_instance.dist,
                            'mu': contact_instance.mu,
                            'includemargin': contact_instance.includemargin,
                            'pos': ','.join(map(str, contact_instance.pos)),
                            'frame': ','.join(map(str, contact_instance.frame.flatten())),
                            'force': ','.join(map(str, force_in_world_frame)),
                            'torque': ','.join(map(str, torque_in_world_frame)),
                            'friction': ','.join(map(str, contact_instance.friction)),
                            'solref': ','.join(map(str, contact_instance.solref)),
                            'solimp': ','.join(map(str, contact_instance.solimp)),
                        }
                        contact_list_serializable.append(flat_contact_info)
                else:
                    flat_contact_info = {
                        'step': step,
                        'geom1': -1,
                        'geom2': -1,
                        'dist': -1,
                        'mu': -1,
                        'includemargin': -1,
                        'pos': -1,
                        'force': -1,
                        'torque': -1,
                        'frame': -1,
                        'friction': -1,
                        'solref': -1,
                        'solimp': -1
                    }
                    contact_list_serializable.append(flat_contact_info)
                
                file_exists = os.path.isfile(centroids_over_time_txt_file)
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=csv_columns)
                    if not file_exists:
                        writer.writeheader()  # Only write header if file doesn't exist
                    writer.writerows(contact_list_serializable)

                # contact_over_time.append(contact_list_serializable)

            # Advance the simulation by one step (periodic or standard)
            if use_periodic and periodic_body_ids:
                _step_periodic(model, data, periodic_box_size, periodic_body_ids)
            else:
                mujoco.mj_step(model, data)

        print(f"Simulation completed after {options['max_steps']} steps.")

    except Exception as e:
        print(f"Error during simulation: {e}")

    finally:
        # save
        import pickle
        with open(f'{output_path}/data.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open(f'{output_path}/model.pickle', 'wb') as f:
            pickle.dump(model, f)

        
        if options["save_all_data"]: # useless... too heavy
            with open(f'{output_path}/data_all.pickle', 'wb') as f:
                pickle.dump(data_all, f)

        # save centroids and orientations
        # centroids_over_time = np.loadtxt(f'{output_path}/centroids_over_time.txt', delimiter=',')
        # orientations_over_time = np.loadtxt(f'{output_path}/orientations_over_time.txt', delimiter=',')
        # time_points = np.loadtxt(f'{output_path}/time_points.txt', delimiter=',')
        # xipos_over_time = np.loadtxt(f'{output_path}/xipos_over_time.txt', delimiter=',')
        # xmat_over_time = np.loadtxt(f'{output_path}/xmat_over_time.txt', delimiter=',')
        # qvel_over_time = np.loadtxt(f'{output_path}/qvel_over_time.txt', delimiter=',')
        # energy_over_time = np.loadtxt(f'{output_path}/energy_over_time.txt', delimiter=',')

        # centroids_over_time = np.array(centroids_over_time)
        # orientations_over_time = np.array(orientations_over_time)
        # time_points = np.array(time_points)
        # xipos_over_time = np.array(xipos_over_time)
        # xmat_over_time = np.array(xmat_over_time)
        # qvel_over_time = np.array(qvel_over_time)
        # energy_over_time = np.array(energy_over_time)

        
        # np.savetxt(f'{output_path}/centroids_over_time.csv', centroids_over_time, delimiter=',')
        # np.savetxt(f'{output_path}/orientations_over_time.csv', orientations_over_time, delimiter=',')
        # np.savetxt(f'{output_path}/time_points.csv', time_points, delimiter=',')
        # np.savetxt(f'{output_path}/xipos_over_time.csv', xipos_over_time, delimiter=',')
        # np.savetxt(f'{output_path}/xmat_over_time.csv', xmat_over_time, delimiter=',')
        # np.savetxt(f'{output_path}/qvel_over_time.csv', qvel_over_time, delimiter=',')

        # compress
        # np.save(f'{output_path}/centroids_over_time.npz', centroids_over_time, allow_pickle=True)

        # np.savez_compressed(f'{output_path}/all_arrays.npz',
        #     centroids_over_time=centroids_over_time,
        #     orientations_over_time=orientations_over_time,
        #     time_points=time_points,
        #     xipos_over_time=xipos_over_time,
        #     xmat_over_time=xmat_over_time,
        #     qvel_over_time=qvel_over_time,
        #     energy_over_time=energy_over_time,
        # )   

        # np.save(f'{output_path}/orientations_over_time.csv', orientations_over_time, delimiter=',')
        # np.save(f'{output_path}/time_points.csv', time_points, delimiter=',')
        # np.save(f'{output_path}/xipos_over_time.csv', xipos_over_time, delimiter=',')
        # np.save(f'{output_path}/xmat_over_time.csv', xmat_over_time, delimiter=',')
        # np.save(f'{output_path}/qvel_over_time.csv', qvel_over_time, delimiter=',')

        # np.savetxt(f'{output_path}/contact_over_time.csv', contact_over_time, delimiter=',')

        # simple pickle contact doesn't work
        # with open(f'{output_path}/contact_over_time.pickle', 'wb') as f:
        #     pickle.dump(contact_over_time, f)

        # save contact info
        # save_contact_info(data)

        # call save function
        return output_path

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation for rod packings.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input x_relaxed.txt file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    
    # Physics parameters
    parser.add_argument("--friction", type=float, default=0.5, help="Friction coefficient")
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep")
    parser.add_argument("--mass", type=float, default=1.0, help="Rod mass")
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Gravity vector")
    
    # Simulation parameters
    parser.add_argument("--steps", type=int, default=10000, help="Number of simulation steps")
    parser.add_argument("--stride", type=int, default=1000, help="Output stride (frames)")
    
    # Initialization
    parser.add_argument("--kick", type=float, default=0.0, help="Initial velocity sigma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Periodic
    parser.add_argument("--periodic", action="store_true", help="Enable periodic boundaries")
    parser.add_argument("--box-size", type=float, nargs=3, default=[2.0, 2.0, 2.0], help="Periodic box size")
    parser.add_argument("--ar", type=float, help="Aspect Ratio of rods")

    args = parser.parse_args()
    
    # Construct options dict expected by run()
    options = {
        "timestep": args.dt,
        "gravity": args.gravity,
        "mass": args.mass,
        "friction": args.friction,
        
        "max_steps": args.steps,
        "intercept": args.stride, # Used as stride in the loop
        
        "random_seed": args.seed,
        "initial_kick": args.kick > 0,
        "random_amplitude": [args.kick]*3 + [args.kick]*3, # Simple isotropic kick
        
        "periodic": args.periodic,
        "periodic_box_size": args.box_size,
        
        # Defaults for things not exposed yet
        "add_ground_plane": False,
        "add_box_boundaries": False,
        "rod_radius": None, # Will infer from file or use default
        "ar": args.ar,
        "rod_length": 1.0,
        
        "save_all_data": False,
        "all_data_interval": 10000
    }
    
    # Ensure output dir exists
    import os
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print(f"Starting MuJoCo simulation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Options: {options}")
    
    try:
        run(file_path=args.input, output_path=args.output, options=options)
        print("Simulation done.")
    except Exception as e:
        print(f"Simulation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
