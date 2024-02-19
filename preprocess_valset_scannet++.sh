#!/bin/bash

# Specify the path to your text file containing the scene IDs
ID_FILE="/home/ml3d/openmask3d/selected_scenes_valset.txt"

# Loop through each line in the file
while IFS= read -r scene_id
do
    # Execute your Python script with the current scene_id from the file
    # Add any other default arguments you need to include
    python /home/ml3d/openmask3d/scannet++_to_openmask.py --scene_id "$scene_id" 

    # If you want to include some echo statements to track the progress, you can uncomment the following line
    echo "Processed scene ID: $scene_id"
done < "$ID_FILE"
