###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage as ndi
from skimage import util
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import os
from utils.util_funcs import relabel_sequentially
from utils.config import Config
import numpy as np
import cv2
import skimage.io as io
from skimage import measure
import shutil
import math
# Use hierarchical clustering to merge 2d segments to 3d segments
# 1. find centers for each 2d segments
# 2. cluster the centers
# 3. merge the segments if the corresponding centers are clustered as one cluster

def slice_merge_AHC(data_name, root_dir, k_list, vox_thresh=0, outlier_thresh=0):
    """function for merging 2d detection results to 3d detection results

    Args:
        data_name ([str]): [name of the dataset]
        src_dir ([str]): [source dir under which all 2d detection slices and scores are list on different volume dir]
        dest_dir ([str]): [destination dir to save merged results]
        k_list ([tuple]): [number of clusters need to try]
        vox_thresh (int, optional): [number of eles in a cluster less than which is considered as outlier]. Defaults to 0.
        outlier_thresh (int, optional): [number of voxels less than which the obj will be revmoed during color coding]. Defaults to 0.
    """
    dest_dir = os.path.join(root_dir, 'AHC')
    savepath = os.path.join(dest_dir, 'seg_results')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    mask_dir = os.path.join(root_dir, 'masks')   # dir containing all 2d detection slices
    score_dir = os.path.join(root_dir, 'scores') # dir containing all confidence scores for each obj for a slice
    vol_names = os.listdir(mask_dir)
    vol_names.sort()
    for vv, vol_name in enumerate(vol_names):
        print('processing volume ', (vv+1))
        img_names = sorted(os.listdir(os.path.join(mask_dir, vol_name)))
        score_names = sorted(os.listdir(os.path.join(score_dir, vol_name)))
        cum_labels = 0
        cents = {}
        mapped = {}
        mapped[0] = 0
        vol = None
        scores_dict = {}
        for index, (img_name, score_name) in enumerate(zip(img_names, score_names)):
            label = io.imread(os.path.join(mask_dir, vol_name, img_name)).astype(np.uint16)
            score = np.load(os.path.join(score_dir, vol_name, score_name))
            shape = label.shape
            if vol is None:
                vol = np.zeros((len(img_names), shape[0], shape[1]), np.uint16)
            # assume the objects are labeled with unique incremental sequential number e.g. [0, 1, 2, 3, 4, 5, 6, 7]
            # label = measure.label(label, connectivity=1).astype(np.uint16)    # relabel changed corresponding between intensity and scores
            max_label = np.max(label)
            label[label != 0] += cum_labels
            # key: mask intensity, value: mask confidence score
            scores_dict.update(dict(zip(np.delete(np.unique(label), 0), score)))

            cum_labels += max_label
            vol[index, :, :] = label

            for ii in range(cum_labels+1-max_label, cum_labels+1):
                center = center_of_mass(label == ii)
                # if np.isnan(center[0]):
                #     continue
                cents[(round(center[0], 2), round(center[1], 2),round(index, 2))] = ii
                mapped[ii] = ii

        # Hierarchical clustering starts here
        # k_list = k_list_meta[data_name]
        k, labels = compare_k_AggClustering(k_list, np.array(list(cents.keys())), outlier_thresh)
        pixel_value = 1
        for key in labels.keys():
            item = labels[key]
            # item = item[np.argsort(item[:,2])]  # sort by z coordinates
            # pixel_value = cents[tuple(item[0])]
            for center in item:
                mapped[cents[tuple(center)]] = pixel_value
            pixel_value += 1

        # print(mapped)
        # Apply mapped values to the volume
        vol = np.vectorize(mapped.get)(vol).astype(np.uint16)
        scores_keys = np.vectorize(mapped.get)(list(scores_dict.keys()))
        scores_values = np.array(list(scores_dict.values()))
        final_scores = []
        for k in np.unique(scores_keys):
            final_scores.append(np.average(scores_values[scores_keys==k]))
        # needs to add code to connect two objs with same intensity value
        vol = same_obj_connection(vol)

        # small object removal
        # cc = measure.label(vol, connectivity=1) # adding this may cause split for an non-connect object, which will increase the number of totoal object in the volume
        cc = relabel_sequentially(vol)
        props = measure.regionprops(cc)
        for jj in range(0, np.amax(cc)):
            if props[jj].area < vox_thresh:
                cc[props[jj].coords[:, 0], props[jj].coords[:, 1],
                    props[jj].coords[:, 2]] = 0
        # vol = measure.label(cc, connectivity=1).astype(np.uint16)
        vol = relabel_sequentially(cc)
        print('number of objects after post-processing: ', len(np.unique(vol))-1)
        # save to individual slices for visualization,
        if not os.path.exists(os.path.join(savepath, vol_name)):
            os.makedirs(os.path.join(savepath, vol_name))
        for zz in range(0, vol.shape[0]):
            cv2.imwrite(os.path.join(savepath, vol_name,
                                     img_names[zz]), vol[zz, :, :])
        scores_dir = os.path.join(dest_dir, 'scores')
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
        np.save(os.path.join(scores_dir, vol_name), final_scores)

        savepath_3d = os.path.join(dest_dir, 'seg_results_3d')
        if not os.path.exists(savepath_3d):
            os.makedirs(savepath_3d)
        io.imsave(os.path.join(savepath_3d, 'vol_' +
                                 str(vv+1).zfill(2)+'.tif'), vol)

        # Color Coding
        # Colormap - Read text files saved from MATLAB
        color_code_path = os.path.join(dest_dir, 'color_coded', vol_name)
        if not os.path.exists(color_code_path):
            os.makedirs(color_code_path)
        cmap = []
        thColormap = 50
        ins = open("./utils/cmap.txt", "r")
        for line in ins:
            line = line.strip().split("\t")
            line2 = [float(n) for n in line]
            line3 = [int(line2[0]), int(line2[1]), int(line2[2])]
            cmap.append(line3)

        ins.close()

        cmap2 = []

        num_colors = 0
        # Dark color removal from colormap
        for i in range(0, len(cmap)):
            if cmap[i][0] > thColormap or cmap[i][1] > thColormap or cmap[i][2] > thColormap:
                cmap2.append(cmap[i])
                num_colors = num_colors + 1

        print("colormap done")
        Z = vol.shape[0]
        Y = vol.shape[1]
        X = vol.shape[2]
        bw3 = util.img_as_ubyte(np.zeros([Z, Y, X, 3]))

        for ii in range(0, Z):
            for jj in range(0, Y):
                for kk in range(0, X):
                    if vol[ii, jj, kk] != 0:
                        bw3[ii, jj, kk, :] = cmap2[(
                            vol[ii, jj, kk]-1) % num_colors]
            cv2.imwrite(os.path.join(color_code_path, "z%04d.png" %
                                     (ii+1)), bw3[ii, :, :, :])

def slice_merge_by_volume_AHC(data_name, masks_list, scores_list, k_list, vox_thresh=0, outlier_thresh=0):
    """same as 'slice_merge_AHC', only difference is input

    Args:
        data_name ([str]): [name of the dataset]
        masks_list ([mask1, mask2,...]): [source dir under which all 2d detection slices and scores are list on different volume dir]
        scores_list ([score1, score2,...]): [destination dir to save merged results]
        k_list ([tuple]): [number of clusters need to try]
        vox_thresh (int, optional): [number of eles in a cluster less than which is considered as outlier]. Defaults to 0.
        outlier_thresh (int, optional): [number of voxels less than which the obj will be revmoed during color coding]. Defaults to 0.
    """
    cum_labels = 0
    cents = {}
    mapped = {}
    mapped[0] = 0
    vol = None
    scores_dict = {}
    for index, (label, score) in enumerate(zip(masks_list, scores_list)):
        shape = label.shape
        if vol is None:
            vol = np.zeros((len(masks_list), shape[0], shape[1]), np.uint16)
        # assume the objects are labeled with unique incremental sequential number e.g. [0, 1, 2, 3, 4, 5, 6, 7]
        # label = measure.label(label, connectivity=1).astype(np.uint16)    # relabel changed corresponding between intensity and scores
        max_label = np.max(label)
        label[label != 0] += cum_labels
        # key: mask intensity, value: mask confidence score
        scores_dict.update(dict(zip(np.delete(np.unique(label), 0), score)))

        cum_labels += max_label
        vol[index, :, :] = label

        for ii in range(cum_labels+1-max_label, cum_labels+1):
            center = center_of_mass(label == ii)
            # if np.isnan(center[0]):
            #     continue
            cents[(round(center[0], 2), round(center[1], 2),round(index, 2))] = ii
            mapped[ii] = ii

    # Hierarchical clustering starts here
    # k_list = k_list_meta[data_name]
    if len(np.array(list(cents.keys()))) <= k_list[0]:  # if the volume containing no objects, return empty vol and score
        return np.zeros(vol.shape, vol.dtype), []
    k, labels = compare_k_AggClustering(k_list, np.array(list(cents.keys())), outlier_thresh)
    pixel_value = 1
    for key in labels.keys():
        item = labels[key]
        # item = item[np.argsort(item[:,2])]  # sort by z coordinates
        # pixel_value = cents[tuple(item[0])]
        for center in item:
            mapped[cents[tuple(center)]] = pixel_value
        pixel_value += 1

    # print(mapped)
    # Apply mapped values to the volume
    vol = np.vectorize(mapped.get)(vol).astype(np.uint16)
    scores_keys = np.vectorize(mapped.get)(list(scores_dict.keys()))
    scores_values = np.array(list(scores_dict.values()))
    final_scores = []
    for k in np.unique(scores_keys):
        final_scores.append(np.average(scores_values[scores_keys==k]))
    # assert len(np.unique(vol))-1==len(final_scores)
    # needs to add code to connect two objs with same intensity value
    vol = same_obj_connection(vol)
    assert np.max(vol) == len(final_scores)
    # small object removal
    # cc = measure.label(vol, connectivity=1) # adding this may cause split for an non-connect object, which will increase the number of totoal object in the volume
    # cc = relabel_sequentially(vol)  # labels might messed up from here, why need this line? why need to remove small obj here
    # props = measure.regionprops(cc)
    # for jj in range(0, np.amax(cc)):
    #     if props[jj].area < vox_thresh:     # need to remove scores as well
    #         cc[props[jj].coords[:, 0], props[jj].coords[:, 1],
    #             props[jj].coords[:, 2]] = 0
    # # vol = measure.label(cc, connectivity=1).astype(np.uint16)
    # vol = relabel_sequentially(cc)
    # print('number of objects after post-processing: ', len(np.unique(vol))-1)

    # Color Coding - why need color code here?
    # Colormap - Read text files saved from MATLAB
    # cmap = []
    # thColormap = 50
    # ins = open("./utils/cmap.txt", "r")
    # for line in ins:
    #     line = line.strip().split("\t")
    #     line2 = [float(n) for n in line]
    #     line3 = [int(line2[0]), int(line2[1]), int(line2[2])]
    #     cmap.append(line3)

    # ins.close()

    # cmap2 = []

    # num_colors = 0
    # # Dark color removal from colormap
    # for i in range(0, len(cmap)):
    #     if cmap[i][0] > thColormap or cmap[i][1] > thColormap or cmap[i][2] > thColormap:
    #         cmap2.append(cmap[i])
    #         num_colors = num_colors + 1

    # print("colormap done")
    # Z = vol.shape[0]
    # Y = vol.shape[1]
    # X = vol.shape[2]
    # bw3 = util.img_as_ubyte(np.zeros([Z, Y, X, 3]))
    # for ii in range(0, Z):
    #     for jj in range(0, Y):
    #         for kk in range(0, X):
    #             if vol[ii, jj, kk] != 0:
    #                 bw3[ii, jj, kk, :] = cmap2[(vol[ii, jj, kk]-1) % num_colors]
    # vol: gray scale volume, bw3: color coded volume, final_scores: confidence scores
    return vol, final_scores


def compare_k_AggClustering(k_list, X, outlier_thresh):
    # to find the best k number of clusters
    # X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    # If a cluster has less than `outlier_thresh` elements, remove this cluster
    if k_list[1] > len(X):
        k_list = (k_list[0], len(X)-1)
    silhouette_list = []

    # the following snipped of code can be parallized
    for p in range(k_list[0], k_list[1]):
        clusterer = AgglomerativeClustering(n_clusters=p, linkage="average")
        clusterer.fit(X)
        # The higher (up to 1) the better
        # print(clusterer.labels_)
        s = round(metrics.silhouette_score(X, clusterer.labels_), 4)
        silhouette_list.append(s)

    # The higher (up to 1) the better
    key = silhouette_list.index(max(silhouette_list))
    k = range(k_list[0],k_list[1]).__getitem__(key)
    # print("Best silhouette =", max(silhouette_list), " for k=", k)
    # check how many clusters with only one point
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='average')
    clusterer.fit(X)
    unique, counts = np.unique(clusterer.labels_, return_counts=True)
    # print(unique)
    # print(counts)

    thre_idx = counts >= outlier_thresh
    counts = counts[thre_idx]
    unique = unique[thre_idx]
    # loop all filtered labels
    average_centers = []

    labels = {}
    for label in unique:
        centers = X[clusterer.labels_ == label]
        labels[label] = centers
    # print('number of cluster after filtering', len(counts))
    # print('number of cluster after filtering', k)
    return len(counts), labels

def same_obj_connection(vol):
    # vol must be in shape (Z, Y, X), Z must be the first axis
    vol = vol.copy()
    labels = np.unique(vol).tolist()
    labels.remove(0)
    for label in labels:
        new_vol = (vol == label)
        cc = measure.label(new_vol, connectivity=1)
        if len(np.unique(cc))>2:   # indicates there are more than 1 object
            idx_z, idx_y, idx_x = np.where(cc!=0)
            missing_idx_z = set(range(min(np.unique(idx_z)), max(np.unique(idx_z)+1))) - set(np.unique(idx_z))
            # to fill in the pixels in a missing slice, need to do intersection of it's neighbors
            # e.g. if slice 8 is missing, need to find intersection of pixels in slice 7 and slice 9
            # e.g. if slice 4 and 5 is missing, need to find intersection of pixels in slice 3 and slice 6
            consecutives = get_consecutives(missing_idx_z)  # there might be errors (empty value) because two disconnected components may lay on same z slice
            for cons in consecutives:
                xy_idx = tuple(zip(idx_y, idx_x))
                upper_pixels = [xy_idx[i] for i in np.where(idx_z == cons[0]-1)[0]]
                lower_pixels = [xy_idx[i] for i in np.where(idx_z == cons[1]+1)[0]]
                filled_pixels = set(upper_pixels).intersection(set(lower_pixels))
                for z in range(cons[0], cons[1]+1):
                    for y, x in filled_pixels:
                        vol[z, y, x] = label
    return vol

def slice_interpolation(vol, conf_vol):
    """Same as same_obj_connection, only difference is input, and do for conf_volume as well"""
    # vol must be in shape (Z, Y, X), Z must be the first axis
    vol = vol.copy()
    labels = np.unique(vol).tolist()
    labels.remove(0)
    for label in labels:
        new_vol = (vol == label)
        cc = measure.label(new_vol, connectivity=1)
        if len(np.unique(cc))>2:   # indicates there are more than 1 object
            idx_z, idx_y, idx_x = np.where(cc!=0)
            assert len(np.unique(conf_vol[cc!=0])) == 1
            prob = np.unique(conf_vol[cc!=0])[0]
            missing_idx_z = set(range(min(np.unique(idx_z)), max(np.unique(idx_z)+1))) - set(np.unique(idx_z))
            # to fill in the pixels in a missing slice, need to do intersection of it's neighbors
            # e.g. if slice 8 is missing, need to find intersection of pixels in slice 7 and slice 9
            # e.g. if slice 4 and 5 is missing, need to find intersection of pixels in slice 3 and slice 6
            consecutives = get_consecutives(missing_idx_z)  # there might be errors (empty value) because two disconnected components may lay on same z slice
            for cons in consecutives:
                xy_idx = tuple(zip(idx_y, idx_x))
                upper_pixels = [xy_idx[i] for i in np.where(idx_z == cons[0]-1)[0]]
                lower_pixels = [xy_idx[i] for i in np.where(idx_z == cons[1]+1)[0]]
                filled_pixels = set(upper_pixels).intersection(set(lower_pixels))
                for z in range(cons[0], cons[1]+1):
                    for y, x in filled_pixels:
                        vol[z, y, x] = label
                        conf_vol[z, y, x] = prob
    return vol

def get_consecutives(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


if __name__ == "__main__":
    data_name = 'sherry_ensemble'
    opt = Config(data_name)
    src_dir = os.path.join(opt.wmf_dir)
    dest_dir = os.path.join(opt.wmf_dir, 'AHC')
    slice_merge_AHC(data_name, src_dir, dest_dir, opt.k_list, opt.voxel_thresh, opt.outlier_thresh)

    # test small_obj_connection
    # vol = io.imread('/data/wu1114/p219/Documents/Detectron2/results/f44_ensemble/weighted_mask_fusion/AHC/seg_results_3d/vol_01.tif')
    # vol = same_obj_connection(vol)
    # io.imsave('tmp.tif', vol)
