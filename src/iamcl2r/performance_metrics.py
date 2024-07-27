from typing import Union, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import faiss

import torch

import logging
logger = logging.getLogger('Performance-Metrics')


def calculate_mAP_gldv2(ranked_gallery_indices, gallery_gts, query_gts, topk):
    retrieved_labels = np.array(gallery_gts)[ranked_gallery_indices]
    num_q = ranked_gallery_indices.shape[0]
    average_precision = np.zeros(num_q, dtype=float)
    for i in range(num_q):
        retrieved_indices = np.where(np.in1d(retrieved_labels[i], np.array(query_gts[i])))[0]
        if retrieved_indices.shape[0] > 0:
            retrieved_indices = np.sort(retrieved_indices)
            gts_all_count = min(len(query_gts[i]), topk)
            for j, index in enumerate(retrieved_indices):
                average_precision[i] += (j + 1) * 1.0 / (index + 1)
            average_precision[i] /= gts_all_count
    return np.mean(average_precision)

# def calculate_mAP_gldv2(ranked_gallery_indices, query_gts, topk):
#     num_q = ranked_gallery_indices.shape[0]
#     average_precision = np.zeros(num_q, dtype=float)
#     for i in range(num_q):
#         retrieved_indices = np.where(np.in1d(ranked_gallery_indices[i], np.array(query_gts[i])))[0]
#         if retrieved_indices.shape[0] > 0:
#             retrieved_indices = np.sort(retrieved_indices)
#             gts_all_count = min(len(query_gts[i]), topk)
#             for j, index in enumerate(retrieved_indices):
#                 average_precision[i] += (j + 1) * 1.0 / (index + 1)
#             average_precision[i] /= gts_all_count
#     return np.mean(average_precision)


def image2template_feature(img_feats=None,  # features of all images
                           templates=None,  # target of features in input 
                          ):
    unique_templates = np.unique(templates)
    unique_subjectids = None

    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        ind_t = np.where(templates == uqt)[0]
        face_norm_feats = img_feats[ind_t]
        template_feats[count_template] = np.mean(face_norm_feats, 0)
        
    logger.info(f'Finish Calculating {count_template} template features.')
    template_norm_feats = template_feats / np.sqrt(
        np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids


# def calculate_rank(query_feats, gallery_feats, topk):
#     logger.info(f"query_feats shape: {query_feats.shape}")
#     logger.info(f"gallery_feats shape: {gallery_feats.shape}")
#     num_q, feat_dim = query_feats.shape

#     logger.info("=> build faiss index")
#     # gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1)[:, np.newaxis]
#     # query_feats = query_feats / np.linalg.norm(query_feats, axis=1)[:, np.newaxis]
#     faiss_index = faiss.IndexFlatIP(feat_dim)
#     # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
#     faiss_index.add(gallery_feats)
#     logger.info("=> begin faiss search")
#     _, ranked_gallery_indices = faiss_index.search(query_feats, topk)
#     return ranked_gallery_indices

def calculate_rank(query_feats, gallery_feats, topk, identical=False):
    logger.info(f"query_feats shape: {query_feats.shape}")
    logger.info(f"gallery_feats shape: {gallery_feats.shape}")
    num_q, feat_dim = query_feats.shape

    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1)[:, np.newaxis]
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1)[:, np.newaxis]
    logger.info("=> build faiss index")
    faiss_index = faiss.IndexFlatIP(feat_dim)
    faiss_index.add(gallery_feats)
    logger.info("=> begin faiss search")
    _, ranked_gallery_indices = faiss_index.search(query_feats, topk + (1 if identical else 0))

    if identical:
        ranked_gallery_indices = ranked_gallery_indices[:, 1:]
    return ranked_gallery_indices

def calculate_cmc(ranked_gallery_indices, gallery_gts, query_gts):
    """
    Calculate Cumulative Martching Characteristics for gallery and query features.
    """
    ranked_gallery_indices = ranked_gallery_indices.copy()
    query_gts = query_gts.copy()
    retrieved_pred = np.array(gallery_gts)[ranked_gallery_indices]

    topk_retrieval = (retrieved_pred == query_gts).sum(axis=1).clip(max=1)
    acc = np.sum(topk_retrieval) / topk_retrieval.shape[0]
    return acc


def identification(gallery_feats, gallery_gts, query_feats, query_gts, topk=1, identical=False):
    # https://github.com/TencentARC/OpenCompatible/blob/master/data_loader/GLDv2.py#L129

    # check if torch, if yes convert to numpy
    if isinstance(query_feats, torch.Tensor):
        query_feats = query_feats.cpu().numpy()
    if isinstance(gallery_feats, torch.Tensor):
        gallery_feats = gallery_feats.cpu().numpy()
    if isinstance(query_gts, torch.Tensor):
        query_gts = query_gts.cpu().numpy()
    if isinstance(gallery_gts, torch.Tensor):
        gallery_gts = gallery_gts.cpu().numpy()

    query_gts = np.array(query_gts).reshape(-1, 1)
    gallery_gts = np.array(gallery_gts)

    logger.info("=> calculate rank")
    ranked_gallery_indices = calculate_rank(query_feats, gallery_feats, topk=topk, identical=identical)
    logger.info("=> calculate 1:N search acc")
    cmc_acc = calculate_cmc(ranked_gallery_indices, gallery_gts, query_gts)

    mAP = calculate_mAP_gldv2(ranked_gallery_indices, gallery_gts, query_gts, topk=topk)
    logger.info(f"1:N search acc: {mAP:.4f}")
    return mAP


# compatibility metrics
def create_position_matrix(matrix, **kwargs):
  position = np.zeros_like(matrix, dtype=bool)
  for j in range(matrix.shape[0]):
    for i in range(j + 1, matrix.shape[1]):
      if matrix[i][j] <= matrix[j][j]:
        position[i, j] = True
  return position


def average_compatibility(matrix, position=None, **kwargs):
  steps = matrix.shape[0]
  if position is None:
    position = create_position_matrix(matrix)
  max_ac = (steps * (steps-1)) / 2
  if max_ac < 1:
    max_ac = 1
  ac = max_ac - np.sum(position)
  return (1/max_ac) * ac


def replace_zero_with_nan(matrix, **kwargs):
  idx = np.where(matrix == 0)
  matrix[idx] = np.nan
  return matrix


def average_accuracy(matrix, per_task=False, **kwargs):
  if max(matrix[-1]) < 1:
    matrix = matrix * 100
  copy_matrix = matrix.copy()
  values = [np.nanmean(replace_zero_with_nan(copy_matrix)[:i+1,:i+1]) for i in range(copy_matrix.shape[0])]

  if per_task:
    return values
  else:
    return values[-1]
