###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
from utils.config import Config
from utils.slice_merge.AHC import slice_merge_AHC
from utils.slice_merge.layercake import slice_merge_layercake
from utils.mask_overlay import mask_overlay

def slice_merge_ensembled_model(data_name, method):
    """Chose a data and a method to merge slices
    This method works on the ensembled model

    Args:
        data_name (str): load settings from config base on data_name
        method (str): choose which method to use for slice merging
    """
    config = Config(data_name)
    if method == "AHC":
        # src_dir = os.path.join(opt.wmf_dir, 'masks')
        dest_dir = os.path.join(config.wmf_dir, 'AHC')
        slice_merge_AHC(data_name, config.wmf_dir, config.k_list, config.voxel_thresh, config.outlier_thresh)
        mask_overlay(config.test_img_orig_dir, dest_dir, config.orig_img_size,alpha=0.9)
    elif method == 'layercake':
        src_dir = os.path.join(config.wmf_dir, 'masks')
        dest_dir = os.path.join(config.wmf_dir, 'layercake')
        slice_merge_layercake(src_dir, dest_dir, radius=config.layercake_radius, voxel_thresh=config.voxel_thresh)
        mask_overlay(config.test_img_orig_dir, dest_dir, config.orig_img_size,alpha=0.9)

def slice_merge_without_ensemble_fusion(data_name, method):
    """Chose a data and a method to merge slices
    This method works on all ensemble models

    Args:
        data_name (str): load settings from config base on data_name
        method (str): choose which method to use for slice merging
    """
    config = Config(data_name)
    if method == "AHC":
        ensemble_names = sorted([name for name in os.listdir(config.ensemble_dir) if 'ensemble' in name])
        for i, ensemble_name in enumerate(ensemble_names):
            src_dir = os.path.join(config.ensemble_dir, ensemble_name)
            dest_dir = os.path.join(config.ensemble_dir, ensemble_name, 'AHC')
            slice_merge_AHC(data_name, src_dir, config.k_list, config.voxel_thresh, config.outlier_thresh)
            mask_overlay(config.test_img_orig_dir, dest_dir, config.orig_img_size, alpha=0.9)
    elif method == 'layercake':
        ensemble_names = sorted([name for name in os.listdir(config.ensemble_dir) if 'ensemble' in name])
        for i, ensemble_name in enumerate(ensemble_names):
            src_dir = os.path.join(config.ensemble_dir, ensemble_name, 'masks')
            dest_dir = os.path.join(config.ensemble_dir, ensemble_name, 'layercake')
            slice_merge_layercake(src_dir, dest_dir, radius=config.layercake_radius, voxel_thresh=config.voxel_thresh)
            mask_overlay(config.test_img_orig_dir, dest_dir, config.orig_img_size, alpha=0.9)

# Merge down the fused results from different ensembles
# Two options 1. 3d AHC, 2. Layercake
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, type=str, help='name of the dataset')
    opt = parser.parse_args()
    data_name = opt.data_name
    slice_merge_ensembled_model(data_name, 'AHC')  # merge fused results obtained from ensemble fusion using AHC method
    slice_merge_ensembled_model(data_name, 'layercake')  # merge fused results obtained from ensemble fusion using BS method
    slice_merge_without_ensemble_fusion(data_name, 'AHC')  # merge for each detectors using AHC method
    slice_merge_without_ensemble_fusion(data_name, 'layercake')  # merge for each detectors using BS method
