import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def visualize_data(data):
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(data.shape[0]):
        r1 = data[i,0:3]
        r2 = data[i,3:6]
        ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]])
    ax.axis('equal')
    plt.show()
def read_from_file(file_path):
    dta = np.loadtxt(file_path, delimiter=' ', comments='#')
    return dta
def read_from_q_file(file_path):
    """
    q: x,y,z, 
    """
    def sph2cart(theta, phi, r=1):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)

    def q_to_x(q):
        q = q.reshape((-1, 5))
        x0 = q[:, :3]
        offsets = sph2cart(q[:, 3], q[:, 4])
        x1 = x0 + offsets
        x = np.concatenate([x0, x1], axis=1)
        return x

    q_data = np.loadtxt(file_path, delimiter=' ').reshape(-1, 5)
    x_data = q_to_x(q_data)
    return q_data, x_data

def visualize_from_file(file_path):
    data = read_from_file(file_path)
    visualize_data(data)
def convert_to_cq_format(file_path,base_axis=np.array([0,0,1])):
    """
    Convert the two points data to the format that can be used in the centroid-quaternion format.
    """
    two_points_data = read_from_file(file_path)
    n = two_points_data.shape[0]
    cq_data = np.zeros((n,7))
    for i in range(n):
        r1 = two_points_data[i,0:3]
        r2 = two_points_data[i,3:6]
        centroid = (r1+r2)/2
        orientation = (r2-r1)/np.linalg.norm(r2-r1)
        # get the quaternion representing rotation from base_axis to orientation
        q = np.zeros(4)
        q[0] = np.cos(np.arccos(np.dot(base_axis,orientation))/2)
        q[1:] = np.cross(base_axis,orientation)*np.sin(np.arccos(np.dot(base_axis,orientation))/2)
        cq_data[i,0:3] = centroid
        cq_data[i,3:] = q
    return cq_data

def convert_to_two_points_format(cq_data, base_axis=np.array([0, 0, 1])):
    """
    Convert centroid-quaternion data to two-points format.
    
    Parameters:
        cq_data (np.ndarray): Array of shape (n, 7), where each row contains [x, y, z, qx, qy, qz, qw].
        base_axis (np.ndarray): The vector to rotate by the quaternion (default is [0, 0, 1]).
    
    Returns:
        np.ndarray: Array of shape (n, 6), where each row contains [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z].
    """
    n = cq_data.shape[0]
    two_points_data = np.zeros((n, 6))
    
    for i in range(n):
        centroid = cq_data[i, :3]
        quaternion = cq_data[i, 3:]  # Quaternion [qx, qy, qz, qw]
        
        # Rotate base_axis by the quaternion
        rotation = R.from_quat(quaternion)
        orientation = rotation.apply(base_axis)
        
        # Calculate the two points
        r1 = centroid - orientation
        r2 = centroid + orientation
        two_points_data[i, :3] = r1
        two_points_data[i, 3:] = r2

    return two_points_data

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


def test_two_points_data():
    # file_path = '/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/26,27,6/MaxEnt26,27,6-N200-AR0100-Scale1.txt'
    file_path = '/Users/yeonsu/GitHub/entanglement-optimization/results/89,32,178/2024-12-23_22_EntangledRelaxedPacking-N0100-AR0100-Scale1/x_relaxed.txt'
    # visualize_from_file(file_path)
    cq_data = convert_to_cq_format(file_path)
    two_points_format = convert_to_two_points_format(cq_data)
    visualize_data(two_points_format)

def test_q_data():
    file_path = "/Users/yeonsu/GitHub/entanglement-optimization/results/89,32,178/2024-12-23_22_EntangledRelaxedPacking-N0100-AR0100-Scale1/q_relaxed.txt"
    q_data, x_data = read_from_q_file(file_path)

    print(q_data.shape)
    print(x_data.shape)

    visualize_data(x_data)

def generate_filled_hcp_cone(R, H_layers, R_base):
    D = 2 * R
    h = np.sqrt(2/3) * D  # vertical distance between HCP layers
    coords = []
    base_layer_indices = []

    a = D  # horizontal spacing in x
    b = np.sqrt(3) * R  # vertical spacing in y

    dx = R  # horizontal shift for B layer
    dy = (1 * np.sqrt(3) / 3) * R  # vertical shift for B layer

    for l in range(H_layers):
        z = l * h
        r_max = R_base * (1 - 0.2*l / H_layers)

        for i in range(-100, 101):
            for j in range(-100, 101):
                # base A layer
                x = i * a + (a / 2 if j % 2 else 0)
                y = j * b

                # apply B-layer shift
                if l % 2 == 1:
                    x += dx
                    y += dy

                if np.sqrt(x**2 + y**2) <= r_max:
                    coords.append((x, y, z))
                    if l == 0:
                        base_layer_indices.append(len(coords) - 1)

    return np.array(coords), base_layer_indices

def random_balls(num_balls, radius, box_halfsize, target_volume_fraction, random_seed=1):
    """
    Generate random non-overlapping balls in a 3D box.

    Parameters:
    - num_balls: int, number of balls to generate
    - radius: float, radius of each ball
    - box_halfsize: float, half-length of the cubic box
    - random_seed: int, seed for reproducibility

    Returns:
    - ball_coords: (num_balls, 3) array of ball centers
    """
    np.random.seed(random_seed)
    ball_coords = []

    max_attempts = 10000000
    attempts = 0

    area = (2 * box_halfsize) ** 2  # Area of the base of the box
    volume_per_particle = (4/3) * np.pi * (radius ** 3)  # Volume of a single ball

    box_height = (num_balls * volume_per_particle)/(target_volume_fraction * area)
    

    while len(ball_coords) < num_balls and attempts < max_attempts:
        candidate = np.random.uniform(-box_halfsize + radius, box_halfsize - radius, size=3)
        # z axis is free
        candidate[2] = np.random.uniform(0 + radius, box_height)  # Random z-coordinate

        # if not ball_coords or np.all(np.linalg.norm(candidate - np.array(ball_coords), axis=1) >= 2 * radius):
        if not ball_coords or np.all(np.linalg.norm(candidate - np.array(ball_coords), axis=1) >= 2 * radius):
            ball_coords.append(candidate)
        attempts += 1

        if attempts % 1000 == 0:
            print(f"Attempts: {attempts}, Balls generated: {len(ball_coords)}")

    if len(ball_coords) < num_balls:
        raise RuntimeError("Failed to generate non-overlapping balls after many attempts.")
    
    print("Random balls generated successfully.")
    ball_coords = np.array(ball_coords)
    ball_coords[:,2] -= ball_coords[:,2].min()
    return ball_coords

def read_positions_at_last_frame(folder_path):
    import os
    from glob import glob
    from pathlib import Path

    # Find the time_points.txt file in the specified path
    file_path = glob(os.path.join(folder_path, '**', 'xipos_over_time.txt'), recursive=True)
    output_path = Path(file_path[0]).parent if file_path else None
    xipos_file_path = os.path.join(output_path, 'xipos_over_time.txt')
    if not output_path or not os.path.exists(xipos_file_path):
        print(f"File not found: {xipos_file_path}")
        return
    else:
        print(f"File found: {xipos_file_path}")
    xipos_over_time = np.loadtxt(xipos_file_path, delimiter=',', skiprows=1)
    pos_at_last_frame = xipos_over_time[-1].reshape(-1, 3)[1:, :]  # Exclude the first row (time point)

    return pos_at_last_frame

if __name__ == "__main__":
    
    radius = 0.03
    H_layers = 10
    R_base = radius*10
    ball_coords, base_layer_indices = generate_filled_hcp_cone(radius, H_layers, R_base)

    import matplotlib.pyplot as plt
    fig,ax= plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(*ball_coords.T, s=10, color='blue')    
    plt.show()

    # num_rods = 100
    # radius = 0.03
    # box_halfsize = 0.03 * 10  # Half the size of the cubic box
    # ball_coords = random_balls(num_rods, radius, box_halfsize, 0.3)
    
    # import matplotlib.pyplot as plt

    # fig,ax= plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(*ball_coords.T, s=10, color='blue')
    # plt.show()

    # test_two_points_data()

    # ball_coords, base_layer_indices = generate_filled_hcp_cone(0.03,10,0.03*20)
    # print("Number of balls:", len(ball_coords))

    # pos = read_positions_at_last_frame('/Users/yeonsu/Downloads/20250528-1644_RUN_faster_mu0.4')
    # import matplotlib.pyplot as plt
    # fig,ax= plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(*pos.T, s=10, color='blue')
    # plt.show()

    # np.count_nonzero(pos[:,2] < 1e-3)


