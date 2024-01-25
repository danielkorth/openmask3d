import os
import shutil
import argparse
import json
import numpy as np
from PIL import Image

def copy_files(source, destination):
    """
    Copy files from source to destination.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    for file_name in os.listdir(source):
        full_file_name = os.path.join(source, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination)


def extract_aligned_pose(json_file, output_directory):
    """
    Extract aligned pose data from a JSON file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    for frame, details in data.items():
        aligned_pose = details.get('aligned_pose', [])
        frame_number = frame.replace('frame_', '').lstrip('0')
        if not frame_number:
            frame_number = '0'

        if not int(frame_number) % 15 == 0:
            continue
        else:
            frame_number = int(frame_number) // 15


        # invert pose
        aligned_pose = np.array(aligned_pose)
        aligned_pose = np.linalg.inv(aligned_pose)


        with open(f'{output_directory}/{frame_number}.txt', 'w') as output_file:
            for row in aligned_pose:
                line = ' '.join(map(str, row))
                output_file.write(line + '\n')

def rename_files(directory):
    """
    Rename files in a directory based on a specific criterion.
    """
    files = os.listdir(directory)
    filtered_files = sorted(
        [file for file in files if file.startswith('frame_') and int(file[6:12]) % 15 == 0],
        key=lambda x: int(x[6:12])
    )

    suffix = files[0].split(".")[-1]

    for new_name, file in enumerate(filtered_files):
        os.rename(os.path.join(directory, file), os.path.join(directory, f"{new_name}.{suffix}"))

def process_scene(scene_id, resize_factor):
    """
    Process a given scene ID.
    """

    # make scene dir
    os.makedirs(f"/home/ml3d/openmask3d/resources/{scene_id}", exist_ok=True)

    # Move point cloud data
    source_pcl = f"/home/data_hdd/scannet/data/{scene_id}/scans/mesh_aligned_0.05.ply"
    dest_pcl = f"/home/ml3d/openmask3d/resources/{scene_id}/scene_example.ply"
    shutil.copy(source_pcl, dest_pcl)

    # Move iPhone camera images
    source_rgb = f"/home/data_hdd/scannet/data/{scene_id}/iphone/rgb"
    dest_rgb = f"/home/ml3d/openmask3d/resources/{scene_id}/color"
    copy_files(source_rgb, dest_rgb)

    # check if both img dims are divisible by 4, resize if so by bicubic interpolation
    for file_name in os.listdir(dest_rgb):
        full_file_name = os.path.join(dest_rgb, file_name)
        if os.path.isfile(full_file_name):
            img = Image.open(full_file_name)
            width, height = img.size
            if not width % resize_factor == 0 or not height % resize_factor == 0:
                raise ValueError(f"Image dimensions are not divisible by {resize_factor}.")
            else:
                img = img.resize((width // resize_factor, height // resize_factor))
                img.save(full_file_name)

    # Move iPhone depth data
    source_depth = f"/home/data_hdd/scannet/data/{scene_id}/iphone/depth"
    dest_depth = f"/home/ml3d/openmask3d/resources/{scene_id}/depth"
    copy_files(source_depth, dest_depth)

    # Extract intrinsic camera parameters
    intrinsic_file = f"/home/data_hdd/scannet/data/{scene_id}/iphone/pose_intrinsic_imu.json"
    with open(intrinsic_file, 'r') as file:
        data = json.load(file)

    intrinsic_matrix = data['frame_000000']["intrinsic"] # 3x3 matrix
    # to homogeneous coordinates 
    intrinsic_matrix = np.vstack([intrinsic_matrix, [0, 0, 0]])
    intrinsic_matrix = np.hstack([intrinsic_matrix, [[0], [0], [0], [1]]])

    # update the intrinsic matrix to account for downsampling: DONE LATER IN THE ORIGINAL SCRIPT
    # intrinsic_matrix[0, 0] /= resize_factor
    # intrinsic_matrix[1, 1] /= resize_factor
    # intrinsic_matrix[0, 2] /= resize_factor
    # intrinsic_matrix[1, 2] /= resize_factor
    
    intrinsic_dir = f"/home/ml3d/openmask3d/resources/{scene_id}/intrinsic"
    # Save the intrinsic matrix to a file
    os.makedirs(intrinsic_dir, exist_ok=True)
    intrinsic_matrix_file = f"{intrinsic_dir}/intrinsic_color.txt"
    with open(intrinsic_matrix_file, 'w') as file:
        for row in intrinsic_matrix:
            file.write(' '.join(map(str, row)) + '\n')


    # Extract aligned pose
    pose_file = f"/home/data_hdd/scannet/data/{scene_id}/iphone/pose_intrinsic_imu.json"
    output_dir = f"/home/ml3d/openmask3d/resources/{scene_id}/pose"
    os.makedirs(output_dir, exist_ok=True)

    extract_aligned_pose(pose_file, output_dir)

    # Rename files in depth directory
    depth_directory = f'/home/ml3d/openmask3d/resources/{scene_id}/depth'
    rename_files(depth_directory)
    # delate the old frame* files   
    for file in os.listdir(depth_directory):
        if file.startswith('frame_'):
            os.remove(os.path.join(depth_directory, file))

    # Rename files in color directory
    color_directory = f'/home/ml3d/openmask3d/resources/{scene_id}/color'
    rename_files(color_directory)


def main():
    """
    Main function to process the input arguments.
    """
    parser = argparse.ArgumentParser(description='Process ScanNet++ data for OpenMask3D.')
    parser.add_argument('--scene_id', type=str, required=True, help='The Scene ID to process.')    
    parser.add_argument('--resize_factor', type=int, default=4, help='The factor by which to downsample the images.')


    args = parser.parse_args()
    process_scene(args.scene_id, args.resize_factor)

if __name__ == "__main__":
    main()
