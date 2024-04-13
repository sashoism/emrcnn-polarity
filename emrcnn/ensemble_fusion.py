###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
import numpy as np
import skimage.io as io
import random
from multiprocessing import Pool
from utils.weighted_mask_fusion.object_match import object_matching
from utils.config import Config
from utils.util_funcs import relabel_sequentially


def ensemble_fusion(data_name, root_dir, M):
    '''
    data_name: name of the dataset that defined in utils.config, used to retrieve original images and draw masks in `visualize` folder
    root_dir: detection results from each model is under this dir
    M: number of models in an ensemble used
    '''
    dest_dir = os.path.join(root_dir, 'weighted_mask_fusion')
    config = Config(data_name)
    orig_img_names = sorted(os.listdir(config.test_img_dir))

    ensembles = sorted([f for f in os.listdir(root_dir) if 'ensemble' in f])
    ensembles = random.sample(ensembles, M)
    # deterine how many volumes needs to be processed
    vol_names = sorted(os.listdir(os.path.join(root_dir, ensembles[0], 'masks')))
    # for each volume
    for v, vol_name in enumerate(vol_names):
        print('v  num:', v)
        if isinstance(config.z, int):
            z_num = config.z
        elif isinstance(config.z, list):
            z_num = config.z[v]
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
        # results = [object_matching([imgs_list[1], scores_list[1]])] # this line is for debugging
        pool = Pool(8)
        results = pool.map(object_matching, list(zip(imgs_list, scores_list)))
        # save fused results to dir `weighted_mask_fusion`
        if not os.path.exists(os.path.join(dest_dir, 'masks', vol_name)):
            os.makedirs(os.path.join(dest_dir, 'masks', vol_name))
        if not os.path.exists(os.path.join(dest_dir, 'scores', vol_name)):
            os.makedirs(os.path.join(dest_dir, 'scores', vol_name))
        if not os.path.exists(os.path.join(dest_dir, 'visualize', vol_name)):
            os.makedirs(os.path.join(dest_dir, 'visualize', vol_name))
        if not os.path.exists(os.path.join(dest_dir, 'cluster', vol_name)):
            os.makedirs(os.path.join(dest_dir, 'cluster', vol_name))
        for i in range(len(results)):
            new_coords, new_scores, cluster_img = results[i]
            new_mask = np.zeros(imgs_list[0][0].shape, dtype=np.uint16)
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
            # save mask
            io.imsave(os.path.join(dest_dir, 'masks', vol_name, img_names[i]), new_mask, check_contrast=False)
            # save score
            np.save(os.path.join(dest_dir, 'scores', vol_name, score_names[i]), new_scores) # bug fix: new_socre -> new_scores
            # save cluster
            io.imsave(os.path.join(dest_dir, 'cluster',vol_name, img_names[i]), cluster_img, check_contrast=False)
            # save visualize
            # orig_img = cv2.imread(os.path.join(opt.test_img_dir, orig_vol[i]))
            # vis_img = draw_on_img(orig_img, new_mask, new_scores, opt)
            # io.imsave(os.path.join(dest_dir, 'visualize', vol_name, img_names[i]), vis_img)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, type=str, help='name of the dataset')
    opt = parser.parse_args()
    data_name = opt.data_name
    config = Config(data_name)
    ensemble_fusion(config.data_name, config.ensemble_dir, config.ensemble)
