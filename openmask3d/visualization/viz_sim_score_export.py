import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
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
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


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
        if self.embedding_model.__class__.__name__ == "DinoV2Model":
            # do not unsqueeze the image for DINO V2
            pass
        else:   
            image_input_processed = image_input_processed.unsqueeze(0)
        
        with torch.no_grad():
            image_embedding_normalized = self.embedding_model.encode_image(image_input_processed).float().cpu()
        return image_embedding_normalized.squeeze().numpy()


    def compute_similarity_scores(self, mask_features, text_query, remove_outliers=False, agg_fct='mean'):
        dist_matrix = list()
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            # TODO check whether axis=1 is correct, or whether it should be axis=2
            mask_norm = np.linalg.norm(mask_emb, axis=1, keepdims=True)
            if mask_norm.sum() < 0.001:
                continue
            with np.errstate(divide='ignore'):
                normalized_emb = (mask_emb/mask_norm)
                # filter NaN
                normalized_emb = normalized_emb[(mask_norm > 0.001).squeeze()]

            def cosine_similarity_metric(u, v):
                return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

            # for distances debugging
            distances = pdist(normalized_emb, metric=cosine_similarity_metric)
            dist_matrix.append(squareform(distances).reshape(-1))

            if remove_outliers:
                # TODO hyperparameter tuning
                min_samples = (normalized_emb.shape[0] // 2) + 1
                eps = 0.1
                # use DBSCAN to remove outliers with cosine similarity distance

                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(squareform(distances))
                # get biggest cluster
                cluster_mask = clustering.labels_ != -1 
                if cluster_mask.sum() != 0:
                    normalized_emb = normalized_emb[cluster_mask]
            if agg_fct == 'mean': 
                scores[mask_idx] = np.nanmean(normalized_emb@text_emb)
            elif agg_fct == 'max':
                scores[mask_idx] = np.nanmax(normalized_emb@text_emb)
            else:
                raise ValueError("provide a valid aggregation function.")
            
        np.save("/home/ml3d/openmask3d_daniel/distances/dist.npz", np.concatenate(dist_matrix))

        return scores
    
    def compute_similarity_scores_for_images(self, mask_features, image_query, agg_fct="mean", **kwargs):
        img_emb = self.get_image_embedding(image_query)
        if agg_fct == "mean":
            print("Aggregating features using mean of crops features.")
            mask_features = mask_features.mean(axis=1)
        
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
        if True: # if true, normalize the scores based on the min. and max. scores for this scene
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
    # with crops
    experiment_path = args.experiment_path
    path_pred_masks = f"{experiment_path}/scene_example_masks.pt"
    path_openmask3d_features = f"{experiment_path}/scene_example_openmask3d_features.npy"
    config_file = f"{experiment_path}/hydra_outputs/mask_features_computation/.hydra/config.yaml"
    ctx = OmegaConf.load(config_file)
    path_scene_pcd = ctx.data.point_cloud_path


    embedding_name: Embedders = ctx.external.embedding_model if ctx.external.embedding_model else 'clip'
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
    if args.text and not args.image_path:
        query_text = args.text
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text, remove_outliers=args.remove_outliers, agg_fct=args.agg_fct)
    elif args.image_path:
        query_image = Image.open(args.image_path)
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores_for_images(openmask3d_features, query_image,
                                                                                                remove_outliers=args.remove_outliers, agg_fct=args.agg_fct)
    else:
        raise ValueError("Please provide either a text or an image query")
    

    # --------------------------------
    # Visualize the similarity scores
    # --------------------------------
    # get the per-point heatmap colors for the similarity scores
    per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(per_mask_query_sim_scores, 
                                                                                                pred_masks,
                                                                                                # normalize_based_on_current_min_max=True
                                                                                                ) # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity

    # visualize the scene with the similarity heatmap
    scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
    scene_pcd_w_sim_colors.points = scene_pcd.points
    scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(per_point_similarity_colors)
    scene_pcd_w_sim_colors.estimate_normals()

    # downsampled = o3d.geometry.voxel_down_sample(scene_pcd_w_sim_colors, 0.01)
    print("COMPUTATION DONE. LOOK!!!")
    # o3d.visualization.draw_geometries([scene_pcd_w_sim_colors.voxel_down_sample(0.01)])
    # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file

    scene_id = ctx.output.output_directory.split('/')[-1]
    views = 'diverse-views' if ctx.openmask3d.preselection > 5 else 'base-views'
    if args.image_path:
        o3d.io.write_point_cloud("data/scene_{}_{}_{}_{}_agg-{}_remove-out-{}_image_query.ply".format(scene_id, '_'.join(args.image_path.split('/')[-1].split('.')[0].split(' ')), embedding_name, views, args.agg_fct, args.remove_outliers), scene_pcd_w_sim_colors)
    else:
        o3d.io.write_point_cloud("data/scene_{}_{}_{}_{}_agg-{}_remove-out-{}.ply".format(scene_id, '_'.join(query_text.split(' ')), embedding_name, views, args.agg_fct, args.remove_outliers), scene_pcd_w_sim_colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Similarity Computation')
    parser.add_argument('-e', '--experiment_path', type=str, help='Path to the experiment folder')
    parser.add_argument('-t', '--text', type=str, default='table', help='Query text')
    parser.add_argument('-i', '--image_path', type=str, help='Query image path')
    parser.add_argument('--remove_outliers', type=str, default='False', help='Whether to remove outliers or not')
    parser.add_argument('--agg_fct', type=str, default='mean', help='Aggregation function for feature embeddings')

    args = parser.parse_args()
    # convert string to boolean
    remove_outliers = args.remove_outliers.lower() == 'true'
    args.remove_outliers = remove_outliers

    main(args)
