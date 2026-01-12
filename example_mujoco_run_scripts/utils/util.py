import mujoco
import pickle
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap, lax
from jax import random

# Helper function to get geom names
def get_geom_name(model,geom_id):
    name_offset = model.name_geomadr[geom_id]
    return model.names[name_offset:].split(b'\0', 1)[0].decode('utf-8')

def get_contact_info(model,data):
    for i in range(data.ncon):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = data.contact[i]
        geom1_name = get_geom_name(model,contact.geom1)
        geom2_name = get_geom_name(model,contact.geom2)

    

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


def read_original_data(original_data_path):
    original_data = read_and_convert_to_co_format(original_data_path) # in centroid-orientation format
    return original_data



def create_mujoco_model_from_file(file_path,options=None):
    if options == None:
        options = {
            "add_ground_plane": False,
            "add_box_boundaries": False
        }

    g = options["gravity"]
    m = options["mass"]
    friction = options["friction"]

    # get AR from file name
    parent_name = options['file_path'].split('/')[-2]
    ar = parent_name.split('-AR')[1].split('-')[0]
    ar = float(ar)
    radius = 1/ar/2
    

    two_points_data = read_from_file(file_path)
    co_data = read_and_convert_to_co_format(file_path) # centroid-orientation format

    xml = f"""
    <mujoco model="particles_in_box">
        <option timestep="{options["timestep"]}" gravity="{g[0]} {g[1]} {g[2]}" integrator="RK4"/>
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

    # TODO: FIGURE OUT BELOW!!
    # for i in range(num_particles):
    #     x,y,z = co_data[i,0],co_data[i,1],co_data[i,2]
    #     u,v,w = co_data[i,3],co_data[i,4],co_data[i,5]
        
    #     # x1,y1,z1,x2,y2,z2 = two_points_data[i]
    #     # (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    #     x1 = x - 0.5*u
    #     y1 = y - 0.5*v
    #     z1 = z - 0.5*w
    #     x2 = x + 0.5*u
    #     y2 = y + 0.5*v
    #     z2 = z + 0.5*w

    #     xml += f"""
    #     <body name="particle{i}" >
    #         <geom name="particle{i}" type="cylinder" size="0.005" mass="{m}" rgba="0.8 0.8 0.8 1" fromto="{x1} {y1} {z1} {x2} {y2} {z2}"/>
    #         <joint type="free"/>
    #     </body>
    #     """
    
    for i in range(num_particles):
        x,y,z = co_data[i,0],co_data[i,1],co_data[i,2]
        # x -= global_centroid[0]
        # y -= global_centroid[1]
        # z -= global_centroid[2]
        u,v,w = co_data[i,3],co_data[i,4],co_data[i,5]
        xml += f"""
        <body name="particle{i}" pos="{x} {y} {z}">
            <geom name="particle{i}" type="cylinder" size="{radius} 0.5" mass="{m}" rgba="0.8 0.8 0.8 1" zaxis="{u} {v} {w}"/>
            <joint type="free"/>
        </body>
        """
    xml += "</worldbody></mujoco>"
    return xml

def create_mujoco_model_from_data(two_points_data,co_data,options=None):
    if options == None:
        options = {
            "add_ground_plane": False,
            "add_box_boundaries": False
        }

    g = options["gravity"]
    m = options["mass"]
    friction = options["friction"]

    # get AR from file name
    parent_name = options['file_path'].split('/')[-2]
    ar = parent_name.split('-AR')[1].split('-')[0]
    ar = float(ar)
    radius = 1/ar/2
    
    

    # two_points_data = read_from_file(file_path)
    # co_data = read_and_convert_to_co_format(file_path) # centroid-orientation format

    xml = f"""
    <mujoco model="particles_in_box">
        <option timestep="{options["timestep"]}" gravity="{g[0]} {g[1]} {g[2]}" integrator="RK4"/>
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

    # print(np.max(two_points_data[:,:3] - global_centroid,axis=0))
    # print(np.min(two_points_data[:,:3] - global_centroid,axis=0))

    # print(np.max(two_points_data[:,3:] - global_centroid,axis=0))
    # print(np.min(two_points_data[:,3:] - global_centroid,axis=0))

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

    # TODO: FIGURE OUT BELOW!!
    # for i in range(num_particles):
    #     x,y,z = co_data[i,0],co_data[i,1],co_data[i,2]
    #     u,v,w = co_data[i,3],co_data[i,4],co_data[i,5]
        
    #     # x1,y1,z1,x2,y2,z2 = two_points_data[i]
    #     # (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    #     x1 = x - 0.5*u
    #     y1 = y - 0.5*v
    #     z1 = z - 0.5*w
    #     x2 = x + 0.5*u
    #     y2 = y + 0.5*v
    #     z2 = z + 0.5*w

    #     xml += f"""
    #     <body name="particle{i}" >
    #         <geom name="particle{i}" type="cylinder" size="0.005" mass="{m}" rgba="0.8 0.8 0.8 1" fromto="{x1} {y1} {z1} {x2} {y2} {z2}"/>
    #         <joint type="free"/>
    #     </body>
    #     """
    
    for i in range(num_particles):
        x,y,z = co_data[i,0],co_data[i,1],co_data[i,2]
        x -= global_centroid[0]
        y -= global_centroid[1]
        z -= global_centroid[2]
        u,v,w = co_data[i,3],co_data[i,4],co_data[i,5]
        xml += f"""
        <body name="particle{i}" pos="{x} {y} {z}">
            <geom name="particle{i}" type="cylinder" size="{radius} 0.5" mass="{m}" rgba="0.8 0.8 0.8 1" zaxis="{u} {v} {w}"/>
            <joint type="free"/>
        </body>
        """
    xml += "</worldbody></mujoco>"
    return xml



def get_clusters(contact_ij,num_rods):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_rods))
    G.add_edges_from(contact_ij)
    clusters = list(nx.connected_components(G))
    num_clusters = len(clusters)
    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_sizes = np.array(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    return clusters, num_clusters, cluster_sizes, max_cluster_size



def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    # p_i = jnp.array([x_i, y_i, z_i])
    # p_j = jnp.array([x_j, y_j, z_j])
    # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    # u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    # p_ii = p_i + l*u_i
    # p_jj = p_j + l*u_j

    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    tol = 0.

    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    
    # def case1(D1,D2,S1,S2,R):
    #     u = 0.
    #     t = fixbound(S1 / D1)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case2(D1,D2,S1,S2,R):
    #     t = 0
    #     u = fixbound(-S2 / D2)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case3(D1,D2,S1,S2,R):
    #     t = 0.
    #     u = 0.
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case4(D1,D2,S1,S2,R):
    #     t = 0.
    #     u = -S2 / D2
    #     uf = fixbound(u)
    #     t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # def case5(D1,D2,S1,S2,R):
    #     t = fixbound((S1 * D2 - S2 * R) / den)
    #     u = (t * R - S2) / D2
    #     uf = fixbound(u)        
    #     t, u = lax.cond(uf != u, lambda _: (fixbound((uf * R + S1) / D1), uf), lambda _: (t, u), None)
    #     return compute_distance(d1, d2, d12, t, u)
    
    # # lax.cond((D1 == 0) & (D2 == 0) , lambda _: 0., lambda _: 0., None)

    # dist = lax.cond((D1 != 0.) & (D2 == 0.),
    #                     lambda _: case1(D1,D2,S1,S2,R),
    #                     lambda _: 0.,
    #                     None)
    
    # dist = lax.cond((D1 == 0.) & (D2 != 0.),
    #                     lambda _: case2(D1,D2,S1,S2,R),
    #                     lambda _: 0.,
    #                     None)
    
    # dist = lax.cond((D1 == 0.) & (D2 == 0.),
    #                     lambda _: case3(D1,D2,S1,S2,R),
    #                     lambda _: 0.,
    #                     None)
    
    # dist = lax.cond((D1 != 0.) & (D2 != 0.) & (den == 0.), # parallel
    #                     lambda _: case4(D1,D2,S1,S2,R),
    #                     lambda _: case5(D1,D2,S1,S2,R),
    #                     None)
    
    return dist

@jit
def acn_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: compute_linking_number_cartesian(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

@jit
def dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):
    # vmap over the index pairs to compute the distance for each unique pair
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)


def analyze_pairwise_dist_entanglement(nodes_at_ith_frame,num_rods):
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = nodes_at_ith_frame.reshape(-1,6)[:,:3]
    r2 = nodes_at_ith_frame.reshape(-1,6)[:,3:]
    pairwise_acn = acn_over_ij(r1,r2, i_indices, j_indices)
    pairwise_dist = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
    dist = jnp.min(pairwise_dist)
    entanglement = jnp.sum(jnp.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    return dist,entanglement,pairwise_dist, pairwise_acn