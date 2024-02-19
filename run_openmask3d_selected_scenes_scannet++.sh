#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -e

# Check if all required command line arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 SCENE_FILE_PATH EMBEDDING_MODEL PRESELECTION " >&2
    echo "Example: $0 selected_scenes.txt clip 5" >&2
    exit 1
fi

SCENE_FILE_PATH="$1"
SCANNET_INSTANCE_GT_DIR="/home/scannet/instance_gt" 

# SET THE CONFIGS
OM_top_k=5
OM_multi_level_expansion_ratio=0.1
OM_num_of_levels=3
OM_vis_threshold=0.1
OM_frequency=1
OM_num_random_rounds=10
OM_num_selected_points=5
OM_preselection=$2

EXT_sam_checkpoint='../resources/sam_vit_h_4b8939.pth'
EXT_sam_model_type='vit_h'
EXT_embedding_model=$3

INF_remove_outliers=false
INF_agg_fct='mean'

# Create a parent output directory for all experiments
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
PARENT_OUTPUT_DIRECTORY="$(pwd)/output/${TIMESTAMP}-experiments"

# Ensure the output directory exists
mkdir -p "${PARENT_OUTPUT_DIRECTORY}"

# Iterate over each line in the scene file
while IFS= read -r SCENE_ID
do
    echo "[INFO] Processing scene: ${SCENE_ID}"

    # Setup directories and paths for the current scene
    SCENE_DIR="$(pwd)/resources/${SCENE_ID}"
    SCENE_POSE_DIR="${SCENE_DIR}/pose"
    SCENE_INTRINSIC_PATH="${SCENE_DIR}/intrinsic/intrinsic_color.txt"
    SCENE_INTRINSIC_RESOLUTION="[1440,1920]"
    SCENE_PLY_PATH="${SCENE_DIR}/scene_example.ply"
    SCENE_COLOR_IMG_DIR="${SCENE_DIR}/color"
    SCENE_DEPTH_IMG_DIR="${SCENE_DIR}/depth"
    IMG_EXTENSION=".jpg"
    DEPTH_EXTENSION=".png"
    DEPTH_SCALE=1000
    MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_model.ckpt"
    SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"
    EXPERIMENT_NAME="${SCENE_ID}"
    OUTPUT_FOLDER_DIRECTORY="${PARENT_OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}"
    SAVE_VISUALIZATIONS=true
    SAVE_CROPS=true
    OPTIMIZE_GPU_USAGE=false

    # Ensure the output directory for the current scene exists
    mkdir -p "${OUTPUT_FOLDER_DIRECTORY}"

    cd openmask3d

    # Compute class agnostic masks
    echo "[INFO] Extracting class agnostic masks for ${SCENE_ID}..."
    python class_agnostic_mask_computation/get_masks_single_scene.py \
    general.experiment_name=${EXPERIMENT_NAME} \
    general.checkpoint=${MASK_MODULE_CKPT_PATH} \
    general.train_mode=false \
    data.test_mode=test \
    model.num_queries=120 \
    general.use_dbscan=true \
    general.dbscan_eps=0.95 \
    general.save_visualizations=${SAVE_VISUALIZATIONS} \
    general.scene_path=${SCENE_PLY_PATH} \
    general.mask_save_dir="${OUTPUT_FOLDER_DIRECTORY}" \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation"
    echo "[INFO] Mask computation done for ${SCENE_ID}!"

    # Compute mask features
    MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
    MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
    SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}"
    echo "[INFO] Masks saved to ${SCENE_MASK_PATH} for ${SCENE_ID}."

    echo "[INFO] Computing mask features for ${SCENE_ID}..."
    python compute_features_single_scene.py \
    data.masks.masks_path=${SCENE_MASK_PATH} \
    data.camera.poses_path=${SCENE_POSE_DIR} \
    data.camera.intrinsic_path=${SCENE_INTRINSIC_PATH} \
    data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
    data.depths.depths_path=${SCENE_DEPTH_IMG_DIR} \
    data.depths.depth_scale=${DEPTH_SCALE} \
    data.depths.depths_ext=${DEPTH_EXTENSION} \
    data.images.images_path=${SCENE_COLOR_IMG_DIR} \
    data.images.images_ext=${IMG_EXTENSION} \
    data.point_cloud_path=${SCENE_PLY_PATH} \
    output.output_directory=${OUTPUT_FOLDER_DIRECTORY} \
    output.save_crops=${SAVE_CROPS} \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation" \
    external.sam_checkpoint=${SAM_CKPT_PATH} \
    gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
    openmask3d.top_k=${OM_top_k} \
    openmask3d.multi_level_expansion_ratio=${OM_multi_level_expansion_ratio} \
    openmask3d.num_of_levels=${OM_num_of_levels} \
    openmask3d.vis_threshold=${OM_vis_threshold} \
    openmask3d.frequency=${OM_frequency} \
    openmask3d.num_random_rounds=${OM_num_random_rounds} \
    openmask3d.num_selected_points=${OM_num_selected_points} \
    external.sam_checkpoint=${EXT_sam_checkpoint} \
    external.sam_model_type=${EXT_sam_model_type} \
    external.embedding_model=${EXT_embedding_model} \
    openmask3d.preselection=$2

    echo "[INFO] Feature computation done for ${SCENE_ID}!"

    # Go back to the original directory
    cd ..

done < "$SCENE_FILE_PATH"


mkdir -p "${PARENT_OUTPUT_DIRECTORY}/evaluation"

# 3. Evaluate for closed-set 3D semantic instance segmentation
python evaluation/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--pred_dir=${PARENT_OUTPUT_DIRECTORY} \
--embedding_name=${EXT_embedding_model} \
--scene_list_file=$1 \
--output_dir=${PARENT_OUTPUT_DIRECTORY}/evaluation \
--agg_fct=${INF_agg_fct} \
--remove_outliers=${INF_remove_outliers} \
