# BASE/CLIP
# EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-47-experiments
# DIV_VIEWS/CLIP
# EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-53-experiments
# BASE/SIGLIP
# EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-08-experiments
# DIV_VIEWS/SIGLIP
# EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments
# BASE/DINOv2
EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-07-09-30-30-experiments
# DIV_VIEWS/DINOv2
# EXPERIMENT := /home/ml3d/openmask3d_daniel/output/2024-02-07-09-21-50-experiments

CLASS := "a tea package"
REMOVE_OUTLIERS := false
AGG_FCT = "mean"

# Visualizations for Modification #4
viz_dino:
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

viz_dino_image:
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-i /home/ml3d/openmask3d_daniel/IMG_7323.png \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

# Visualization for Modification #2
viz_experiments_1vs1:
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-t $(CLASS) \
															--remove_outliers false \
															--agg_fct 'max'
														
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-t $(CLASS) \
															--remove_outliers false \
															--agg_fct "mean" 

	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-t $(CLASS) \
															--remove_outliers true \
															--agg_fct "max" 


# Visualize across all 6 scenes
viz_experiments:
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/45d2e33be1 \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 
														
	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/75d29d69b8 \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/104acbf7d2 \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/c50d2d1d42 \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/f9f95681fd \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 

	python openmask3d/visualization/viz_sim_score_export.py -e $(EXPERIMENT)/41b00feddb \
															-t $(CLASS) \
															--remove_outliers $(REMOVE_OUTLIERS) \
															--agg_fct $(AGG_FCT) 


# Quantitative Evaluation
run_eval_presentation:
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-47-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-47-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-47-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-47-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-53-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-53-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-53-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-06-22-51-53-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-08-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-08-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-08-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-08-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-07-07-03-02-experiments true mean 5

run_eval_scannet++_40:
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-10-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-10-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-10-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-10-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-41-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-41-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-41-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-13-16-20-41-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-43-10-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-43-10-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-43-10-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-43-10-experiments true mean 5

	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-45-03-experiments false mean 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-45-03-experiments false max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-45-03-experiments true max 5
	bash run_evaluation_scannet++.sh /home/ml3d/openmask3d_daniel/output/2024-02-15-06-45-03-experiments true mean 5

# Run Experiments
experiments_cuda0:
	bash run_openmask3d_selected_scenes_scannet++.sh scannet++_val_scenes.txt 5 clip 
	bash run_openmask3d_selected_scenes_scannet++.sh scannet++_val_scenes.txt 5 siglip
	# bash run_openmask3d_selected_scenes_scannet++.sh selected_scenes.txt 5 dinov2

experiments_cuda1:
	bash run_openmask3d_selected_scenes_scannet++.sh scannet++_val_scenes.txt 10 clip
	bash run_openmask3d_selected_scenes_scannet++.sh scannet++_val_scenes.txt 10 siglip
	# bash run_openmask3d_selected_scenes_scannet++.sh selected_scenes.txt 10 dinov2