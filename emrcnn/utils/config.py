###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
class Config():
    # backbone networks will be randomly initilized for each detector in an ensemble
    backbone_files = [
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    ]

    def __init__(self, data_name):
        self.data_name = data_name
        self.project_dir = os.environ["PROJECT_DIR"]
        self.ensemble = int(os.environ["ENSEMBLES"])

        # The following are settings for training immu data
        # If you want to train your own data, start a new "if" statement and add these variables
        if data_name == "immu_ensemble":
            # where to read json format ground truth information
            self.train_json_dir = os.path.join(self.project_dir, "dataset/immu/train/json")
            # where to read training images
            self.train_img_dir = os.path.join(self.project_dir, "dataset/immu/train/syn")
            # where to read testing image while testing
            self.test_img_dir = os.path.join(self.project_dir, "dataset/immu/test1/real")
            # where to read original testing image while testing
            self.test_img_orig_dir = os.path.join(self.project_dir, "dataset/immu/test1/original")
            # where to read ground truth masks during testing
            self.gt_mask_dir = os.path.join(self.project_dir, "dataset/immu/test1/gt")
            # where to save results: root directory of results for this data
            self.ensemble_dir = os.path.join(self.project_dir, "results/immu_ensemble")
            # where to save results for weighted-masks-fusion
            self.wmf_dir = os.path.join(self.project_dir, "results/immu_ensemble/weighted_mask_fusion")

            self.label_name = "immu"    # the label that print on the detection result for mask rcnn
            self.orig_img_size = 128    # original image size
            self.v = 16                 # number of subvolumes
            self.z = 32                 # number of slice for each volume
            self.k_list = (5, 50)       # number of clusters need to try
            self.outlier_thresh = 1     # number of eles in a cluster less than which is considered as outlier
            self.voxel_thresh = 200     # number of voxels less than which the obj will be revmoed during color coding
            self.layercake_radius = 8   # Euclidea distance for BS method in slice merging

        elif data_name == "cellpol":
            self.train_json_dir = os.path.join(self.project_dir, "dataset/cellpol/train/json")
            # where to read training images
            self.train_img_dir = os.path.join(self.project_dir, "dataset/cellpol/train/real")
            # where to read testing image while testing
            self.test_img_dir = os.path.join(self.project_dir, "dataset/cellpol/test/real")
            # where to read original testing image while testing
            self.test_img_orig_dir = os.path.join(self.project_dir, "dataset/cellpol/test/original")
            # where to read ground truth masks during testing
            self.gt_mask_dir = os.path.join(self.project_dir, "dataset/cellpol/test/gt")
            # where to save results: root directory of results for this data
            self.ensemble_dir = os.path.join(self.project_dir, "results/cellpol")
            # where to save results for weighted-masks-fusion
            self.wmf_dir = os.path.join(self.project_dir, "results/cellpol/weighted_mask_fusion")

            self.label_name = "cellpol" # the label that print on the detection result for mask rcnn
            self.orig_img_size = 64     # original image size
            self.v = 16                 # number of subvolumes
            self.z = 64                 # number of slice for each volume
            self.k_list = (5, 50)       # number of clusters need to try
            self.outlier_thresh = 1     # number of eles in a cluster less than which is considered as outlier
            self.voxel_thresh = 200     # number of voxels less than which the obj will be revmoed during color coding
            self.layercake_radius = 8   # Euclidea distance for BS method in slice merging