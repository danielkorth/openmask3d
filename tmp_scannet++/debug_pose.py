import open3d as o3d
import numpy as np

def create_camera_frustum(width, height, fx, fy, cx, cy, pose, scale=1):
    """
    Create a frustum for visualization.
    width, height: Dimensions of the image plane
    fx, fy: Focal lengths
    cx, cy: Principal point
    pose: Camera pose (4x4 matrix)
    scale: Scale of the frustum
    """
    # Create frustum points
    frustum_points = np.array([
        [0, 0, 0, 1],  # Camera center
        [width, 0, fx*scale, 1],
        [0, height, fy*scale, 1],
        [width, height, (fx+fy)/2*scale, 1],
        [0, 0, (fx+fy)/2*scale, 1]
    ])

    frustum_points[:, :3] -= np.array([cx, cy, 0])  # Move principal point to origin
    frustum_points[:, :3] /= np.array([fx, fy, (fx+fy)/2])  # Normalize by focal lengths
    frustum_points[:, :3] *= scale  # Scale frustum


    pose = np.linalg.inv(pose)  # Invert pose matrix
    # Transform points with camera pose
    frustum_points_transformed = (pose @ frustum_points.T).T[:,:3]

    # Create linesets for visualization
    lines = [[0, 1], [0, 2], [0, 3], [0, 4]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frustum_points_transformed),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

# Define camera parameters
width, height =1920, 1440  # Image plane dimensions

# load intrinsic matrix
intrinsic_matrix_file = "/home/ml3d/openmask3d/resources/41b00feddb/intrinsic/intrinsic_color.txt"
with open(intrinsic_matrix_file, 'r') as file:
    intrinsic_matrix = np.array([list(map(float, line.split())) for line in file.readlines()])

fx = intrinsic_matrix[0,0]
fy = intrinsic_matrix[1,1]
cx = intrinsic_matrix[0,2]
cy = intrinsic_matrix[1,2]


# load camera pose
pose_file = "/home/ml3d/openmask3d/resources/41b00feddb/pose/474.txt"
with open(pose_file, 'r') as file:
    pose = np.array([list(map(float, line.split())) for line in file.readlines()])

# Create frustum
frustum = create_camera_frustum(width, height, fx, fy, cx, cy, pose, scale=1)

# Load a point cloud (replace with your point cloud file)
point_cloud = o3d.io.read_point_cloud("/home/ml3d/openmask3d/resources/41b00feddb/scene_example.ply")

print("Point cloud has", len(point_cloud.points), "points")

# Visualize
o3d.visualization.draw_geometries([point_cloud.voxel_down_sample(0.01),frustum])
