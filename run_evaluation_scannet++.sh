# parse arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 PARENT_OUTPUT_DIRECTORY INF_remove_outliers INF_agg_fct TOPK" >&2
    echo "Example: $0 /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments true mean 5" >&2
    exit 1
fi

PARENT_OUTPUT_DIRECTORY=$1 
SCANNET_INSTANCE_GT_DIR="/home/scannet/instance_gt"
INF_remove_outliers=$2 
INF_agg_fct=$3 
TOPK=$4

OUT_NAME=${PARENT_OUTPUT_DIRECTORY}/evaluation/eval_results_top${TOPK}_${INF_agg_fct}_outliers-${INF_remove_outliers}.txt

mkdir -p "${PARENT_OUTPUT_DIRECTORY}/evaluation"

# 3. Evaluate for closed-set 3D semantic instance segmentation
python openmask3d/evaluation/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--pred_dir=${PARENT_OUTPUT_DIRECTORY} \
--scene_list_file="/home/ml3d/openmask3d_daniel/scannet++_val_scenes.txt" \
--output_dir=${PARENT_OUTPUT_DIRECTORY}/evaluation \
--agg_fct=${INF_agg_fct} \
--remove_outliers=${INF_remove_outliers} \
--topk=${TOPK} \
> ${OUT_NAME} 2>&1
