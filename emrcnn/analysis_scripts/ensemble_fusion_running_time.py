###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
import numpy as np
import skimage.io as io
from multiprocessing import Pool
from utils.weighted_mask_fusion.weighted_mask_fusion import weighted_mask_fusion
from utils.weighted_mask_fusion.object_match import object_matching
from utils.config import Config
from utils.vis import draw_on_img
from utils.config import Config
import time

# similar to "ensemble_fusion.py", just use for checking running time

def ensemble_fusion(data_name, root_dir):
    '''
    data_name: name of the dataset that defined in utils.config, used to retrieve original images and draw masks in `visualize` folder
    root_dir: detection results from each model is under this dir
    '''
    dest_dir = os.path.join(root_dir, 'weighted_mask_fusion')
    opt = Config(data_name)
    orig_img_names = sorted(os.listdir(opt.test_img_dir))

    ensembles = sorted([f for f in os.listdir(root_dir) if 'ensemble' in f])
    # deterine how many volumes needs to be processed
    vol_names = sorted(os.listdir(os.path.join(root_dir, ensembles[0], 'masks')))
    # for each volume
    for v, vol_name in enumerate(vol_names):
        if isinstance(opt.z, int):
            z_num = opt.z
        elif isinstance(opt.z, list):
            z_num = opt.z[v]
        orig_vol = orig_img_names[:z_num]
        del orig_img_names[:z_num]
        # read all ensemble detections for this volume
        img_names = sorted(os.listdir(os.path.join(
            root_dir, ensembles[0], 'masks', vol_name)))
        score_names = sorted(os.listdir(os.path.join(
            root_dir, ensembles[0], 'scores', vol_name)))
        imgs_list = []
        scores_list = []
        for img_name, score_name in zip(img_names, score_names):
            imgs = []   # images needs to be fused, same slice from different models
            scores = []  # scores needs to be fused
            for ensemble in ensembles:
                imgs.append(io.imread(os.path.join(
                    root_dir, ensemble, 'masks', vol_name, img_name)))
                scores.append(np.load(os.path.join(
                    root_dir, ensemble, 'scores', vol_name, score_name)))

            imgs_list.append(imgs)
            scores_list.append(scores)
        # parallel computing, fusing different slices of different ensembles parallely
        pool = Pool(16)
        # results = [object_matching([imgs_list[54], scores_list[54]])] # this line is for debugging
        results = pool.map(object_matching, list(zip(imgs_list, scores_list)))
        # save fused results to dir `weighted_mask_fusion`
        break

# make sure to comment all image writing code to reduce the running time
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default="immu_ensemble", type=str, help='name of the dataset')
    opt = parser.parse_args()
    data_name = opt.data_name
    opt = Config(data_name)
    start = time.time()
    ensemble_fusion(data_name, opt.ensemble_dir)
    print('weighted mask fusion time is:', (time.time() - start)/16)