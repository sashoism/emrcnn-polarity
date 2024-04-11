###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import numpy as np
import os
from utils.vis import draw_cluster

def weighted_mask_fusion(obj_coords, scores_all, labels, centers, shape):
    '''
    fuse the masks in `obj_coords` based on score of each mask in `scores_all`
    'labels' indicates which masks to fuse
    '''
    # idx = np.argsort(labels)
    # labels = np.array(labels)[idx]
    scores_all = np.array(scores_all)
    obj_coords = np.array(obj_coords, dtype='object')
    new_scores = []
    fused_obj_coords = []
    prob_mask_all = np.zeros(shape)  # prob mask for all objects
    for i in np.unique(labels):
        matched_idx = (labels==i)
        matched_scores = scores_all[matched_idx]
        matched_obj_coords = obj_coords[matched_idx]
        # fuse the detection masks based on the score
        # masks are given by coordinates
        prob_mask = np.zeros(shape) # prob mask for same object
        for score, coords in zip(matched_scores, matched_obj_coords):
            prob_mask[coords[0], coords[1]] += score
            prob_mask_all[coords[0], coords[1]] += score

        # prob_mask[prob_mask>=sum(matched_scores)/2] = 1
        # prob_mask[prob_mask<sum(matched_scores)/2] = 0
        fused_obj_coords.append(np.where(prob_mask >= sum(matched_scores)/2))
        # calculate new scores for fused object
        new_scores.append(np.mean(matched_scores))
    
    # for visualization cluster results
    prob_mask_with_cluster = draw_cluster(prob_mask_all, centers, labels)

    return fused_obj_coords, new_scores, prob_mask_with_cluster
