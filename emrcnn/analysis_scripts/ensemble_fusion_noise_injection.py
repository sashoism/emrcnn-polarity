###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
# Script for robustness analysis
################################################################################################

import os
import numpy as np
import skimage.io as io
import skimage.draw as draw
import random
from multiprocessing import Pool
from utils.weighted_mask_fusion.weighted_mask_fusion import weighted_mask_fusion
from utils.weighted_mask_fusion.object_match import object_matching
from utils.config import Config
from utils.vis import draw_on_img
from utils.config import Config
from utils.util_funcs import relabel_sequentially

# adding False Positive detections on each slice of each model before merging to test the robustness

def noise_injection(imgs_list, scores_list, N=2, radius=8):
    """Adding false positive segments

    Args:
        imgs_list ([ZxMxYxX]): Z: number of slices, M: number of models, YxX: image sizes
        scores_list ([ZxMxO]): Z: number of slices, M: number of models, O: number of objects in this image
        N ([int]): number of false positive segments to be added for each image slice from each model
    """
    for z in range(len(scores_list)):
        for m in range(len(scores_list[z])):
            image = imgs_list[z][m]
            score = scores_list[z][m].tolist()
            # adding false positives here
            h, w = image.shape
            n = 0
            while n<N:
                r_radius,c_radius = (radius, radius)
                r,c = (random.randint(0+r_radius, h-r_radius), random.randint(0+c_radius, w-c_radius))  # center coordinates
                # get the coordinates of the ellipse
                coords = draw.ellipse(r, c, r_radius, c_radius)
                obj_mask = np.zeros((h,w))
                obj_mask[coords] = True
                if np.any(np.logical_and(obj_mask, image==0)==True):
                    # entire false positive obj is not covered by existing obj
                    n+=1
                    image[np.logical_and(obj_mask, image==0)] = np.max(image) + 1
                    score.append(random.uniform(0.7, 1.0))  # add a random score for added noise segment
            # set back
            imgs_list[z][m] = image
            scores_list[z][m] = np.array(score)

    return imgs_list, scores_list

def ensemble_fusion_with_noise_injection(data_name, root_dir):
    '''
    data_name: name of the dataset that defined in utils.config, used to retrieve original images and draw masks in `visualize` folder
    root_dir: detection results from each model is under this dir
    '''
    dest_dir = os.path.join(root_dir, 'weighted_mask_fusion')
    opt = Config(data_name)
    orig_img_names = sorted(os.listdir(opt.test_img_dir))

    ensembles = sorted([f for f in os.listdir(root_dir) if 'ensemble' in f])
    # ensembles = ensembles[:4]
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
        # add False Positive segments to each slice
        imgs_list, scores_list = noise_injection(imgs_list, scores_list)
        # parallel computing, fusing different slices of different ensembles parallely
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
                new_obj_mask[new_coord[0], new_coord[1]] = True
                new_obj_loc = np.logical_and(new_obj_mask, new_mask==0)
                if np.all(new_obj_loc==False):
                    removed_obj_idx.append(j)
                new_mask[new_obj_loc] = j+1
            if len(removed_obj_idx)!=0:
                new_mask = relabel_sequentially(new_mask)
                new_scores = np.delete(new_scores, removed_obj_idx)
            # save mask
            io.imsave(os.path.join(dest_dir, 'masks', vol_name, img_names[i]), new_mask)
            # save score
            np.save(os.path.join(dest_dir, 'scores', vol_name, score_names[i]), new_score)
            # save cluster
            io.imsave(os.path.join(dest_dir, 'cluster',vol_name, img_names[i]), cluster_img)
            # save visualize
            # orig_img = cv2.imread(os.path.join(opt.test_img_dir, orig_vol[i]))
            # vis_img = draw_on_img(orig_img, new_mask, new_scores, opt)
            # io.imsave(os.path.join(dest_dir, 'visualize', vol_name, img_names[i]), vis_img)
 


if __name__ == '__main__':
    data_name = 'immu_ensemble'
    opt = Config(data_name)
    ensemble_fusion_with_noise_injection(data_name, opt.ensemble_dir)
