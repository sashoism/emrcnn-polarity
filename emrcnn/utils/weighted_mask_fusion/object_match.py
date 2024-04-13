###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

# this function is used for finding corresponding associated object from different detection maps
from hashlib import new
import os
import numpy as np
import skimage.io as io
from sklearn import metrics
import pandas as pd
from multiprocessing import Pool
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage.measurements import center_of_mass
from .weighted_mask_fusion import weighted_mask_fusion

def AHC(k_list, X):
    # Given data points X and a list of number of clusters, find optimal number of clusters
    # to find the best k number of clusters
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:
        clusterer = AgglomerativeClustering(n_clusters=p, linkage="average")
        clusterer.fit(X)
        # The higher (up to 1) the better
        s = round(metrics.silhouette_score(X, clusterer.labels_), 4)
        silhouette_list.append(s)

    # The higher (up to 1) the better
    key = silhouette_list.index(max(silhouette_list))
    k = k_list.__getitem__(key)
    # print("Best silhouette =", max(silhouette_list), " for k=", k)
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='average')
    clusterer.fit(X)

    return k, clusterer

def get_k_list(N_min_labels, N_max_labels, n_samples):
    N_min_labels = max(N_min_labels, 2)
    N_max_labels = N_max_labels if N_max_labels <= n_samples-1 else n_samples-1
    return range(N_min_labels, N_max_labels+1)

def compare_k_AggClustering(N_obj_list, N_ensemble, X, obj_coords, scores_all):
    obj_coords = np.array(obj_coords, dtype='object')
    scores_all = np.array(scores_all)
    sigma2 = int(np.std(N_obj_list))
    # Assign (min, max) possible clusters with constraints: 2 <= n_labels <= n_samples - 1
    N_min_labels = min(N_obj_list) - sigma2
    N_max_labels = max(N_obj_list) + len(N_obj_list) + sigma2
    # N_max_labels = N_max_labels if N_max_labels <= len(X)-1 else len(X)-1
    k_list = get_k_list(N_min_labels, N_max_labels, len(X))
    # best k number of clusters
    k, clusterer = AHC(k_list, X)
    # check constraints:
    # constraint: number of elements in a cluster <= number of ensembles, should we consider this?
    cluster_id, counts = np.unique(clusterer.labels_, return_counts=True)
    while np.any(counts <= int(N_ensemble/2)): # if any cluster contains less than int(N_ensemble/2) points
        remove_X_idx = []
        cluster_id_remove = cluster_id[counts<=int(N_ensemble/2)]
        for c_id in cluster_id_remove:
            remove_X_idx.extend(list(np.where(clusterer.labels_ == c_id)[0]))
        # remove from original data
        X = np.delete(X, remove_X_idx, 0)
        obj_coords = np.delete(obj_coords, remove_X_idx, 0)
        scores_all = np.delete(scores_all, remove_X_idx, 0)
        # re-run cluster with optimal cluster number ``k-10,k+10``
        k = k - len(cluster_id_remove)
        if k == 0:
            return [], [], [], []
        # k, clusterer = AHC(range(max(k-10, 2), k+10), X)
        new_k_list = get_k_list(k-10, k+10, len(X))
        k, clusterer = AHC(new_k_list, X)
        cluster_id, counts = np.unique(clusterer.labels_, return_counts=True)

    return clusterer.labels_, X.tolist(), obj_coords.tolist(), scores_all.tolist()

def object_matching(inputs):
    # """
    # masks (list): masks from ensemble models, each obj in the mask marked by a unique intensity
    # scores (list): scores for each mask is a numpy array, the index of the numpy array is corresponding to the object intensity in `masks`
    # """
    masks, scores = inputs
    assert len(masks) == len(scores)    # = number of models
    N_ensemble = len(masks)
    N_obj = []
    scores_all = []
    centers = []
    obj_coords = []
    # obj_counter = 0
    for i in range(N_ensemble):
        # for each detection mask and scores
        mask = masks[i]
        score = scores[i]
        # print(i)
        assert len(np.unique(mask))-1==len(score)
        N_obj.append(len(score))
        for p in np.unique(mask):
            if p == 0:
                continue
            # obj_counter += 1
            scores_all.append(score[p-1])
            # mask[mask==p] = obj_counter
            obj_coords.append(np.where(mask==p))

        centers.extend(center_of_mass(mask, mask, np.unique(mask)[1:]))
    # match the centers ``centers``
    # labels = compare_k_AggClustering(range(max(min(N_obj)-5,0), max(N_obj)+6), np.array(centers))
    fused_obj_coords = []
    new_scores = []
    cluster_img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), np.uint8)
    if len(centers) < 3:  # least 3 samples for clustering
        return fused_obj_coords, new_scores, cluster_img
    
    labels, centers, obj_coords, scores_all = compare_k_AggClustering(N_obj, N_ensemble, np.array(centers), obj_coords, scores_all)
    if len(labels) == 0:    # some centroids might be removed by majority voting
        return fused_obj_coords, new_scores, cluster_img

    fused_obj_coords, new_scores, cluster_img = weighted_mask_fusion(obj_coords, scores_all, labels, np.array(centers), masks[0].shape)
    sort_idx = np.argsort(new_scores)[::-1]
    new_scores = list(np.array(new_scores)[sort_idx])
    fused_obj_coords = list(np.array(fused_obj_coords, dtype='object')[sort_idx])
    return fused_obj_coords, new_scores, cluster_img
