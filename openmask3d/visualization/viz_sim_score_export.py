import numpy as np
from omegaconf import OmegaConf
import open3d as o3d
import torch
import clip
import pdb
import matplotlib.pyplot as plt
from constants import *
from openmask3d.embeddings import CLIPModel, SigLIPModel, DinoV2Model, Embedders
import argparse
from PIL import Image

class QuerySimilarityComputation():
    def __init__(self,embedding_model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.embedding_model = embedding_model

    def get_query_embedding(self, text_query):
        text_input_processed = self.embedding_model.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding_normalized = self.embedding_model.encode_text(text_input_processed).float().cpu()

        return sentence_embedding_normalized.squeeze().numpy()
    
    def get_image_embedding(self, image: Image.Image):
        # unsqueeze the image to add a batch dimension
        image_input_processed = self.embedding_model.preprocess_image(image).to(self.device)
        image_input_processed = image_input_processed.unsqueeze(0)
        with torch.no_grad():
            image_embedding_normalized = self.embedding_model.encode_image(image_input_processed).float().cpu()
        return image_embedding_normalized.squeeze().numpy()


    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            # print("Mask Norm is:", mask_norm)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb
            # if isinstance(self.embedding_model, SigLIPModel):

        return scores
    
    def compute_similarity_scores_for_images(self, mask_features, image_query):
        img_emb = self.get_image_embedding(image_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            # print("Mask Norm is:", mask_norm)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@img_emb

        return scores

    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms>normalize_min_bound
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3))*0 + background_color
        
        for mask_idx, mask in enumerate(masks[::-1, :]):
            # get color from matplotlib colormap
            new_colors[mask>0.5, :] = plt.cm.jet(openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1])[:3]

        return new_colors



def main(args):
    # --------------------------------
    # Set the paths
    # --------------------------------
    experiment_path = "/home/ml3d/openmask3d/output/2024-02-05-15-06-59-experiment"

    path_pred_masks = f"{experiment_path}/scene_example_masks.pt"
    path_openmask3d_features = f"{experiment_path}/scene_example_openmask3d_features.npy"
    config_file = f"{experiment_path}/hydra_outputs/mask_features_computation/.hydra/config.yaml"
    ctx = OmegaConf.load(config_file)
    path_scene_pcd = ctx.data.point_cloud_path


    embedding_name: Embedders = ctx.external.embedding_model
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    if embedding_name == "clip":
        embedding_model = CLIPModel(device=device)
    elif embedding_name == "siglip":
        embedding_model = SigLIPModel(device=device)
    elif embedding_name == "dinov2":
        embedding_model = DinoV2Model(device=device)
    else:
        raise ValueError(f"Unknown embedding model: {embedding_name}")
    
    # --------------------------------
    # Load data
    # --------------------------------
    # load the scene pcd
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    
    # load the predicted masks
    pred_masks = np.asarray(torch.load(path_pred_masks)).T # (num_instances, num_points)

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features) # (num_instances, 768)

    # initialize the query similarity computer
    query_similarity_computer = QuerySimilarityComputation(embedding_model)
    

    # # --------------------------------
    # # Set the query text
    # # --------------------------------
    # query_text = "paper" # change the query text here
    
    # --------------------------------
    # Get the similarity scores
    # --------------------------------
    # create query
    if args.text:
        query_text = args.text
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text)
    elif args.image_path:
        query_image = Image.open(args.image_path)
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores_for_images(openmask3d_features, query_image)
    else:
        raise ValueError("Please provide either a text or an image query")
    

    # --------------------------------
    # Visualize the similarity scores
    # --------------------------------
    # get the per-point heatmap colors for the similarity scores
    per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(per_mask_query_sim_scores, 
                                                                                                pred_masks,
                                                                                                normalize_based_on_current_min_max=True) # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity

    # visualize the scene with the similarity heatmap
    scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
    scene_pcd_w_sim_colors.points = scene_pcd.points
    scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(per_point_similarity_colors)
    scene_pcd_w_sim_colors.estimate_normals()

    # downsampled = o3d.geometry.voxel_down_sample(scene_pcd_w_sim_colors, 0.01)
    o3d.visualization.draw_geometries([scene_pcd_w_sim_colors.voxel_down_sample(0.01)])
    # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
    # o3d.io.write_point_cloud("data/scene_pcd_w_sim_colors_{}.ply".format('_'.join(query_text.split(' '))), scene_pcd_w_sim_colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Similarity Computation')
    parser.add_argument('-e', '--experiment_path', type=str, help='Path to the experiment folder')
    parser.add_argument('-t', '--text', type=str, help='Query text')
    parser.add_argument('-i', '--image_path', type=str, help='Query image path')
    
    args = parser.parse_args()


    main(args)
