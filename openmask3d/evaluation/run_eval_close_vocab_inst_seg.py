import os
import numpy as np
import clip
import glob
from omegaconf import OmegaConf
import torch
import pdb
from eval_semantic_instance import evaluate
from scannet_constants import SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20, CLASS_LABELS_20
from scannet_constants import SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200
from scannet_constants import VALID_CLASS_IDS_PP, CLASS_LABELS_PP
from scipy.spatial.distance import pdist, squareform

import tqdm
import argparse
from openmask3d.embeddings import CLIPModel, SigLIPModel

from sklearn.cluster import DBSCAN

class InstSegEvaluator():
    def __init__(self, dataset_type, embedding_name, sentence_structure):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.embedding_name = embedding_name
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        self.set_label_and_color_mapper(dataset_type)

        if self.embedding_name == 'clip':
            self.embedding_model = CLIPModel(self.device)
        elif self.embedding_name == 'siglip':
            self.embedding_model = SigLIPModel(self.device)
        else:
            raise NotImplementedError

        self.feature_size = self.embedding_model.dim
        self.text_query_embeddings = self.get_text_query_embeddings().numpy() #torch.Size([20, 768])

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):
        if dataset_type == 'scannet':
            label_list = list(CLASS_LABELS_20)
            label_list[-1] = 'other' # replace otherfurniture with other, following OpenScene
        elif dataset_type == 'scannet200':
            label_list = list(CLASS_LABELS_200)
        elif dataset_type == 'scannet++':
            label_list = list(CLASS_LABELS_PP)
        else:
            raise NotImplementedError
        return [sentence_structure.format(label) for label in label_list]

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))
        for label_idx, sentence in enumerate(self.query_sentences):
            #print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = self.embedding_model.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding_normalized = self.embedding_model.encode_text(text_input_processed).float().cpu()

            # sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings
    
    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'scannet':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_20)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_20.get)
        elif dataset_type == 'scannet200':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_200)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_200.get)
        elif dataset_type == 'scannet++':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_PP)}.get)
            # self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_PP.get)

        else:    
            raise NotImplementedError

    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # normalize mask features
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]

        similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(177, 20)
        max_class_similarity_scores = np.max(similarity_scores, axis=1) # does not correspond to class probabilities
        max_ind = np.argmax(similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)
        pred_classes = max_ind_remapped

        return masks, pred_classes, max_class_similarity_scores
    

    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, keep_first=None, remove_outliers=False, agg_fct='mean', topk=5):
        pred_masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        keep_mask = np.asarray([True for el in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False

        mask_norm = np.linalg.norm(mask_features, axis=2, keepdims=True)
        with np.errstate(divide='ignore'):
            mask_features_normalized = (mask_features/mask_norm)

        zeros_mask = np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)
        valid_mask = ~np.all(zeros_mask, axis=-1)

        if remove_outliers:
            # TODO hyperparameter tuning
            eps = 0.1

            def cosine_similarity_metric(u, v):
                return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

            for i in range(len(mask_features_normalized)):
                if valid_mask[i].sum() == 0:
                    continue
                min_samples = (valid_mask[i].sum() // 2) + 1
                actual_dist = pdist(mask_features_normalized[i][valid_mask[i]], metric=cosine_similarity_metric)
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(squareform(actual_dist))

                # get biggest cluster
                cluster_mask = clustering.labels_ == -1 
                # if all labels are minus one, keep all

                if ~np.all(cluster_mask):
                    # assign nan to all other clusters

                    temp = mask_features_normalized[i][valid_mask[i]].copy()
                    temp[cluster_mask] = np.nan
                    mask_features_normalized[i][valid_mask[i]] = temp

        # print(mask_features_normalized.shape, self.text_query_embeddings.T.shape) #(177, 20), (20, 768
        per_class_similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(177, 20)
        if agg_fct == 'mean': 
            per_class_similarity_scores = np.nanmean(per_class_similarity_scores, axis=1)
        elif agg_fct == 'max':
            per_class_similarity_scores = np.nanmax(per_class_similarity_scores, axis=1)
        else:
            raise ValueError("provide a valid aggregation function.")
        # max_ind = np.argmax(per_class_similarity_scores, axis=1)

        top_k_ind = np.argpartition(-per_class_similarity_scores, topk, axis=1)[:, :topk]

        # max_ind_remapped = self.label_mapper(max_ind)
        
        # pred_classes = max_ind_remapped[keep_mask]
        pred_masks = pred_masks[:, keep_mask]
        # pred_scores = np.ones(pred_classes.shape)

        pred_scores = np.ones(top_k_ind[:, 0].shape)
        top_k_pred_classes = self.label_mapper(top_k_ind)
        assert top_k_ind.shape[1] == top_k_pred_classes.shape[1]

        return pred_masks, top_k_pred_classes, pred_scores

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_dir='eval_output'):
        #pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))

        output_file = os.path.join(output_dir, 'eval_result.txt')

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score. 

        return inst_AP

def test_pipeline_full_scannet200(mask_features_dir,
                                    gt_dir,
                                    pred_root_dir,
                                    sentence_structure,
                                    feature_file_template,
                                    dataset_type='scannet200',
                                    embedding_name='clip',
                                    keep_first = None,
                                    scene_list_file='evaluation/val_scenes_scannet200.txt',
                                    masks_template='_masks.pt',
                                    remove_outliers=False,
                                    agg_fct='mean',
                                    topk=5
                         ):


    evaluator = InstSegEvaluator(dataset_type, embedding_name, sentence_structure)
    print('[INFO]', dataset_type, embedding_name, sentence_structure)

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()

    preds = {}

    for scene_name in tqdm.tqdm(scene_names[:]):

        scene_id = scene_name[5:]

        masks_path = os.path.join(pred_root_dir, scene_name, masks_template)
        scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))

        if not os.path.exists(scene_per_mask_feature_path):
            print('--- SKIPPING ---', scene_per_mask_feature_path)
            continue
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(masks_path=masks_path, 
                                                                                               mask_features_path=scene_per_mask_feature_path,
                                                                                               keep_first=keep_first,
                                                                                               remove_outliers=remove_outliers,
                                                                                               agg_fct=agg_fct,
                                                                                               topk=topk
                                                                                               )

        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type, output_dir='eval_output')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='path to directory of GT .txt files')
    parser.add_argument('--pred_dir', type=str, help='path to the saved class agnostic masks')
    # parser.add_argument('--mask_features_dir', type=str, help='path to the saved mask features')
    parser.add_argument('--feature_file_template', type=str, default="{}/scene_example_openmask3d_features.npy")
    parser.add_argument('--sentence_structure', type=str, default="a {} in a scene", help='sentence structure for 3D closed-set evaluation')
    parser.add_argument('--scene_list_file', type=str, default="/home/ml3d/openmask3d/selected_scenes.txt")
    parser.add_argument('--masks_template', type=str, default="scene_example_masks.pt")
    parser.add_argument('--remove_outliers', type=str, default='False', help='Whether to remove outliers or not')
    parser.add_argument('--agg_fct', type=str, default='mean', help='Aggregation function for feature embeddings')
    parser.add_argument('--output_dir', type=str, default='eval_output', help='output directory for the evaluation results')
    parser.add_argument('--topk', type=int, default=5, help='number of masks to keep')
    
    

    opt = parser.parse_args()
    # convert string to boolean
    remove_outliers = opt.remove_outliers.lower() == 'true'
    opt.remove_outliers = remove_outliers

    config_file = glob.glob(f"{opt.pred_dir}/*/hydra_outputs/mask_features_computation/.hydra/config.yaml", recursive=True)[0]
    ctx = OmegaConf.load(config_file)
    opt.embedding_name = ctx.external.embedding_model

    # ScanNet200, "a {} in a scene", all masks are assigned 1.0 as the confidence score
    test_pipeline_full_scannet200(opt.pred_dir, opt.gt_dir, opt.pred_dir, 
                                  opt.sentence_structure, opt.feature_file_template,
                                    dataset_type='scannet++', embedding_name=opt.embedding_name,
                                      keep_first=None, scene_list_file=opt.scene_list_file, masks_template=opt.masks_template,
                                        remove_outliers=opt.remove_outliers, agg_fct=opt.agg_fct, topk=opt.topk)