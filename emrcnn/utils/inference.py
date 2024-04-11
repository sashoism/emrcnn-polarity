###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from utils.config import Config
import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import sys
import time
from multiprocessing import Pool
import math
from tqdm import tqdm
import skimage.io as io
import os, json, cv2, random
import shutil
from utils.util_funcs import relabel_sequentially
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
import skimage.io as io
from utils.weighted_mask_fusion.object_match import object_matching
from utils.slice_merge.AHC import slice_merge_by_volume_AHC
from utils.util_funcs import normalize, get_confidence_vol



def pred_volume_assemble(opt, predictors, volume, block_size):
    # volume = io.imread('/data/wu1114/Documents/dataset/for_testing/wsm_microscopy/test_3Dtif/syn/vol001.tif')
    # volume = np.transpose(volume, (2,0,1))
    # volume = np.repeat(volume[:,:,:,np.newaxis], 3, axis=3)
    # for each slice
    Z, H, W, _ = volume.shape
    fused_masks = []
    fused_scores = []
    for z in tqdm(range(Z)):
        im = volume[z,:,:]
        # im = normalize(im[:,:,0])
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # im = clahe.apply(im)
        # im = np.repeat(im[:,:,np.newaxis], 3, axis=2)

        # if len(np.unique(im)) != 1:
        #     print()
        mask_list = []  # mask for each slice
        score_list = [] # score for each slice
        for pred in predictors:
            outputs = pred(im)
            masks = outputs["instances"].to("cpu").pred_masks
            removed_obj_idx = []    # obj are needs to be removed due to overlapping
            mask_combined = np.zeros((block_size, block_size), dtype=np.uint16)
            for ii in range(0, masks.shape[0]):
                mask = np.uint8(masks[ii,:,:])*255
                # to solve the problem that next object overlay the former object
                if np.all(np.logical_and(mask>0.5, mask_combined==0)==False):
                    removed_obj_idx.append(ii)
                mask_combined[np.logical_and(mask>0.5, mask_combined==0)] = ii+1
            if len(removed_obj_idx)!=0:
                mask_combined = relabel_sequentially(mask_combined)
            score = np.delete(outputs["instances"].to("cpu").scores, removed_obj_idx)
            assert len(np.unique(mask_combined))-1 == len(score)
            mask_list.append(mask_combined)
            score_list.append(score)
        fused_masks.append(mask_list)
        fused_scores.append(score_list)
    
    pool = Pool(8)
    results = pool.map(object_matching, list(zip(fused_masks, fused_scores)))

    # perform weighted masks fusions
    masks_list = []
    scores_list = []
    for i in range(len(results)):
        new_coords, new_scores, cluster_img = results[i]
        new_mask = np.zeros((block_size, block_size), dtype=np.uint16)
        removed_obj_idx = []
        for j, (new_coord, new_score) in enumerate(zip(new_coords, new_scores)):
            new_obj_mask = np.zeros(new_mask.shape)
            new_obj_mask[np.int32(new_coord[0]), np.int32(new_coord[1])] = True
            new_obj_loc = np.logical_and(new_obj_mask, new_mask==0)
            if np.all(new_obj_loc==False):
                removed_obj_idx.append(j)
            new_mask[new_obj_loc] = j+1

        if len(removed_obj_idx)!=0:
            new_mask = relabel_sequentially(new_mask)
            new_scores = np.delete(new_scores, removed_obj_idx)

        masks_list.append(new_mask)
        scores_list.append(new_scores)

    # slice merging starts here
    # 'masks_list': a list of masks for each slice on z direction
    # 'scores_list': a list of scores for each slice on z direction
    seg, final_scores = slice_merge_by_volume_AHC(opt.data_name, masks_list, scores_list, 
                                                (opt.k_min, opt.k_max), opt.voxel_thresh, opt.outlier_thresh)
    conf = get_confidence_vol(seg, final_scores)
    # for i in np.unique(seg): # sanity check
    #     assert len(np.unique(conf[seg==i])) == 1
    return seg, conf