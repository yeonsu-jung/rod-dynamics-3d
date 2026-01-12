# %%
import numpy as np
import pickle
import mujoco
import mujoco.viewer
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from util import get_clusters

import ast
import time
# from util import get_geom_name
from matplotlib import pyplot as plt
# from packing_initialization import read_and_convert_to_co_format
import os

def read_from_file(file_path):
    dta = np.loadtxt(file_path, delimiter=' ')
    return dta

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

def read_data(root_dir):
    data_path = f'{root_dir}/data.pickle'
    model_path = f'{root_dir}/model.pickle'
    all_data_path = f'{root_dir}/data_all.pickle'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if os.path.exists(all_data_path):
        with open(all_data_path, 'rb') as f:
            data_all = pickle.load(f)
    else:
        data_all = []

    return data, model, data_all

def read_original_data(original_data_path):
    original_data = read_and_convert_to_co_format(original_data_path) # in centroid-orientation format
    return original_data


if __name__ == "__main__":
    
    import re


    import os
    
    # Check if we are already in the data directory OR the RunFolder
    # If run locally or in the new hierarchical structure, the file might be right here.
    if os.path.exists('x_relaxed.txt') or os.path.exists('options.txt') or os.path.exists('run.sh') or os.path.exists('options.yml'):
        storage_path = os.getcwd()
        print(f"Detected running in data directory: {storage_path}")
        lab_directory = False # Skip the redirect logic
    else:
        lab_directory = 1

    if lab_directory:
        storage_path = '/n/holylabs/LABS/mahadevan_lab/Users/yjung/maximal-entanglement/'
        current_folder = os.getcwd()
        # copy current folder to storage path
        storage_path = storage_path + current_folder.split('/')[-1]
        print('storage_path', storage_path)
    # else:
    #    storage_path = os.getcwd() (already set above if found)
    
    # find x_related.txt from subfolders
    flag = False
    original_data_path = None
    for subdir, dirs, files in os.walk(storage_path):
        for file in files:
            if file == 'x_relaxed.txt':
                original_data_path = os.path.join(subdir, file)
                root_dir = subdir
                break
        if original_data_path is not None:
            flag = True
            break

    print('original_data_path', original_data_path)


    options_file = f'{root_dir}/options.txt'
    if not os.path.exists(options_file):
        print('No options.txt file found. Skipping this directory.')
        exit()

    # Load text from file
    with open(options_file, 'r') as f:
        text = f.read()

    # Convert text to dictionary
    options = ast.literal_eval(text)

    original_data = read_original_data(original_data_path) # in centroid-orientation format
    num_rods = original_data.shape[0]

    meta_dir = options['file_path']
    random_keys = meta_dir.split('/')[-3]
    # print(random_keys)

    # print(meta_dir)
    # pattern = r"keys([0-9,]+)_N([0-9]+)_mu([0-9.]+)_AR([0-9]+)_A([0-9.]+)"
    # match = re.search(pattern, meta_dir)
    # # parse meta dir
    # # keys, N, mu, AR, A
    # if match:
    #     keys = match.group(1)   # '6,7,8'
    #     N = int(match.group(2)) # 200
    #     mu = float(match.group(3)) # 0.05
    #     AR = int(match.group(4)) # 10
    #     A = float(match.group(5)) # 0.01
        
    #     print('keys: ', keys)
    #     print('N: ', N)
    #     print('mu: ', mu)
    #     print('AR: ', AR)
    #     print('A', A)

    # ar = float(ar)
    # ar = options['AR']
    
    try:
        # robust parsing
        # ...AR10-Scale... -> 10
        # ...AR10.txt -> 10
        part = meta_dir.split('AR')[1]
        # split by common delimiters
        import re
        # take leading digits/dots
        match = re.search(r"^[0-9.]+", part)
        if match:
             ar = float(match.group(0))
        else:
             # fallback to simple split if regex fails for some reason
             ar = float(part.split('-')[0].split('.')[0].split('_')[0])
    except Exception as e:
        print(f"Error parsing AR: {e}")
        ar = 100.0 # fallback default? or re-raise
        
    radius = 1/ar/2

    # exp_id = meta_dir.split('/')[-1]
    exp_id = f"keys{random_keys}_N{num_rods}_mu{options['friction']:.2f}_AR{ar:.0f}_A{options['random_amplitude'][0]:.3f}"
    print('exp_id', exp_id)

    model_file = f'{root_dir}/model.pickle'
    data_file = f'{root_dir}/data.pickle'

    # Load the mujoco model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    # Load the mujoco data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    


    # import pickle
    import pickle


    # with open(f'{root_dir}/contact_over_time.pickle', 'rb') as f:

    # with open(f'{root_dir}/contact_over_time.pickle', 'rb') as f:
    #     contact_over_time = pickle.load(f)

    import pandas as pd
    df = pd.read_csv(f'{root_dir}/rod_contact_info.csv')

    steps = sorted(df['step'].unique())  # ensure order

    contact_over_time = []
    for step in steps:
        contacts = df[df['step'] == step]

        # Reconstruct dictionaries per contact
        contact_list = []
        for _, row in contacts.iterrows():
            contact = {
                'geom1': int(row['geom1']),
                'geom2': int(row['geom2']),
                'dist': float(row['dist']),
                'pos': [float(x) for x in row['pos'].split(',')],
                'frame': [float(x) for x in row['frame'].split(',')],
            }
            contact_list.append(contact)

        contact_over_time.append(contact_list)

    contacts = contact_over_time[-1]

    contact_ij = []
    for i in range(len(contacts)):
        contact = contacts[i]
        # print(contact['geom1'],contact['geom2'])
        contact_ij.append([contact['geom1'],contact['geom2']])
    contact_ij = np.array(contact_ij)

    print(len(contact_over_time))
    
    
    clusters, num_clusters, cluster_sizes, max_cluster_size = get_clusters(contact_ij,num_rods)
    print('max cluster size', max_cluster_size)
    print('num pairs', num_rods)

    # npz_file = f'{root_dir}/all_arrays.npz'
    # if not os.path.exists(npz_file):
    #     print('No all_arrays.npz file found. Skipping this directory.')
    #     exit()

    # all_arrays = np.load(npz_file)
    # time_points = all_arrays['time_points']
    # xipos_over_time = all_arrays['xipos_over_time']
    # xmat_over_time = all_arrays['xmat_over_time']
    # centroids_over_time = all_arrays['centroids_over_time']
    # orientations_over_time = all_arrays['orientations_over_time']
    # qvel_over_time = all_arrays['qvel_over_time']
    # energy_over_time = all_arrays['energy_over_time']

    centroids_over_time = np.loadtxt(f'{root_dir}/centroids_over_time.txt', delimiter=',')
    orientations_over_time = np.loadtxt(f'{root_dir}/orientations_over_time.txt', delimiter=',')
    time_points = np.loadtxt(f'{root_dir}/time_points.txt', delimiter=',')
    xipos_over_time = np.loadtxt(f'{root_dir}/xipos_over_time.txt', delimiter=',')
    xmat_over_time = np.loadtxt(f'{root_dir}/xmat_over_time.txt', delimiter=',')
    qvel_over_time = np.loadtxt(f'{root_dir}/qvel_over_time.txt', delimiter=',')
    energy_over_time = np.loadtxt(f'{root_dir}/energy_over_time.txt', delimiter=',')

    time_points = jnp.array(time_points)
    xipos_over_time = jnp.array(xipos_over_time)
    xmat_over_time = jnp.array(xmat_over_time)
    centroids_over_time = jnp.array(centroids_over_time)
    orientations_over_time = jnp.array(orientations_over_time)
    qvel_over_time = jnp.array(qvel_over_time)

    # plt.plot(energy_over_time[1:,1])
    # tmp = np.max(energy_over_time[:,1])*1.1
    # plt.ylim([0,tmp])

    # and sliding ratio
    # and entanglement
    # and fraction of contact network
    xipos_over_time = xipos_over_time[:,3:]
    xmat_over_time = xmat_over_time[:,9:]
    
    xipos_over_time = xipos_over_time.reshape(-1,num_rods,3)
    xmat_over_time = xmat_over_time.reshape(-1,num_rods,3,3)
    original_orientations = original_data[:,3:]

    from util import analyze_pairwise_dist_entanglement

    sliding_over_time = []
    v12_n_over_time = []
    v12_t_over_time = []
    entanglment_over_time = []
    num_contacts_over_time = []
    max_cluster_size_over_time = []
    distance_over_time = []
    

    for i_frame in range(xipos_over_time.shape[0]):
        xipos = xipos_over_time[i_frame]
        ximat = xmat_over_time[i_frame]
        
        # orientations_array = orientations_over_time[i_frame].reshape(-1,3,3)
        # centroids_array = centroids_over_time[i_frame].reshape(-1,3)

        orientations_array = ximat.reshape(-1,3,3)
        centroids_array = xipos.reshape(-1,3)

        # Ensure the batch sizes match
        assert orientations_array.shape[0] == original_orientations.shape[0], "Batch sizes must match!"

        # Perform the matrix multiplication
        cylinder_axes = np.einsum('nij,nj->ni', orientations_array.reshape(-1, 3, 3), original_orientations)

        centroids_array = np.array(centroids_array)
        orientations_array = np.array(orientations_array)

        r1 = centroids_array - 0.5 * cylinder_axes
        r2 = centroids_array + 0.5 * cylinder_axes

        two_points_data = np.concatenate((r1,r2),axis=1) # nrod * 1 * 6
        dist,entanglement,pairwise_dist, pairwise_acn = analyze_pairwise_dist_entanglement(two_points_data,num_rods)
        entanglment_over_time.append(entanglement)

        qvel = qvel_over_time[i_frame].copy().reshape(-1,6)
        contacts = contact_over_time[i_frame]
        ncon = len(contacts)
        sliding_ratio_list = []

        v12_n_list = []
        v12_t_list = []

        
        num_contacts_over_time.append(ncon)
        contact_ij = []
        
        for i in range(ncon):
            if contacts[i]['geom1'] == -1:
                continue

            contact_ij.append([contacts[i]['geom1'],contacts[i]['geom2']])

            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            # TODO: check above statement

            # contact = data.contact[i]
            # print('contact', i)
            # print('dist', contact.dist)

            geom1_idx = contacts[i]['geom1']+1 # why?
            geom2_idx = contacts[i]['geom2']+1
            contact_distance = contacts[i]['dist']
            
            # contact between particles
            r1_i = r1[geom1_idx-1,:]
            r2_i = r2[geom1_idx-1,:]

            r1_j = r1[geom2_idx-1,:]
            r2_j = r2[geom2_idx-1,:]
            
            c1 = centroids_array[geom1_idx-1]
            c2 = centroids_array[geom2_idx-1]
            d1 = contacts[i]['pos'] - centroids_array[geom1_idx-1]
            d2 = contacts[i]['pos'] - centroids_array[geom2_idx-1]

            v1 = qvel[geom1_idx-1,0:3] + np.cross(qvel[geom1_idx-1,3:6],d1)
            v2 = qvel[geom2_idx-1,0:3] + np.cross(qvel[geom2_idx-1,3:6],d2)

            # normal contact velocity
            v_n = np.dot(v1,contacts[i]['frame'][0:3])
            

            v1 = qvel[geom1_idx-1,0:3] + np.cross(qvel[geom1_idx-1,3:6],d1)
            v2 = qvel[geom2_idx-1,0:3] + np.cross(qvel[geom2_idx-1,3:6],d2)
            v12 = v1 - v2
            normal = contacts[i]['frame'][0:3]

            v1_n = np.dot(v1,normal)*v1/np.linalg.norm(v1)
            v1_t = v1 - v1_n
            
            v2_n = np.dot(v2,normal)*v2/np.linalg.norm(v2)
            v2_t = v2 - v2_n

            v12_n = np.dot(v12,normal)*v12/np.linalg.norm(v12)
            v12_t = v12 - v12_n

            # sliding_ratio_list.append(np.linalg.norm(v12_t)/np.linalg.norm(v12_n))
            # sliding_ratio_list.append(np.linalg.norm(v1_t)/np.linalg.norm(v1_n))
            sliding_ratio_list.append(np.linalg.norm(v12_t/v12_n))
            v12_n_list.append(np.linalg.norm(v12_n))
            v12_t_list.append(np.linalg.norm(v12_t))

        clusters, num_clusters, cluster_sizes, max_cluster_size = get_clusters(contact_ij,num_rods)
        max_cluster_size_over_time.append(max_cluster_size)
        sliding_over_time.append(np.sum(sliding_ratio_list))
        v12_n_over_time.append(np.sum(v12_n_list))
        v12_t_over_time.append(np.sum(v12_t_list))

    max_cluster_size_over_time = np.array(max_cluster_size_over_time)
    sliding_over_time = np.array(sliding_over_time)
    entanglment_over_time = np.array(entanglment_over_time)
    v12_t_over_time = np.array(v12_t_over_time)
    v12_n_over_time = np.array(v12_n_over_time)

    
    # what to save?
    # where to save?

    # save to npz file
    npz_file = f'{root_dir}/all_results.npz'
    np.savez(npz_file,
            time_points=time_points,
            num_contacts_over_time=num_contacts_over_time,
            energy_over_time=energy_over_time,
            entanglment_over_time=entanglment_over_time,
            sliding_over_time=sliding_over_time,
            v12_t_over_time=v12_t_over_time,
            v12_n_over_time=v12_n_over_time,
            max_cluster_size=max_cluster_size_over_time
    )
    

# %% visual

    num_frames = xipos_over_time.shape[0]
    num_key_frames = 30
    skip = max(1, num_frames // num_key_frames)

    
    plt.close('all')
    k = 0
    for i_frame in range(1,xipos_over_time.shape[0],skip):
        fig,ax=plt.subplots(figsize=(5,5),subplot_kw={'projection':'3d'})
        xipos = xipos_over_time[i_frame]
        ximat = xmat_over_time[i_frame]

        orientations_array = ximat.reshape(-1,3,3)
        centroids_array = xipos.reshape(-1,3)

        # Ensure the batch sizes match
        assert orientations_array.shape[0] == original_orientations.shape[0], "Batch sizes must match!"

        # Perform the matrix multiplication
        cylinder_axes = np.einsum('nij,nj->ni', orientations_array.reshape(-1, 3, 3), original_orientations)

        centroids_array = np.array(centroids_array)
        orientations_array = np.array(orientations_array)

        r1 = centroids_array - 0.5 * cylinder_axes
        r2 = centroids_array + 0.5 * cylinder_axes

        for i in range(num_rods):
            # plot the cylinder
            ax.plot([r1[i,0],r2[i,0]],[r1[i,1],r2[i,1]],[r1[i,2],r2[i,2]],linewidth=2)
            
        
        plt.axis('equal')
        # axis
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([-2,2])
            
        plt.savefig(f'frame_{k:04d}.png')
        plt.close('all')
        k = k + 1

    
    import subprocess
    subprocess.run(["ffmpeg", "-y","-framerate", "30", "-i", "frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{exp_id}.mp4"])

    # remove all png files
    for i in range(num_key_frames+1):
        os.remove(f'frame_{i:04d}.png')
# %%
