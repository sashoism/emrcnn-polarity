###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import numpy as np
from scipy.ndimage.interpolation import zoom
from skimage import measure
from utils.CC2 import CC2
from tqdm import tqdm
import skimage.io as io
def split_intensities(img):
    # given an image with multiple unique pixel intensities represent different object
    # return a NxHxW bool mask where N is the number of intensities (excluding background)
    img = img.copy()
    mask = np.zeros((len(np.unique(img))-1, img.shape[0], img.shape[1]), dtype=bool)
    for i in np.unique(img):
        if i==0:
            continue
        mask[i-1, img==i] = True
    
    return mask

def relabel_sequentially(vol):
    """relabel objects in a volume, this function does not consider connected components,
    instead, it only re-assign the labels and make all labels sequentially
    e.g. [0,1,2,4,5,6,9,10] -> [0,1,2,3,4,5,6,7]

    Args:
        vol (ndarray): ndarray that contains different intensities
    """
    vol = vol.copy()
    i = 1
    for s in np.unique(vol):
        if s == 0:
            continue
        vol[vol==s] = i
        i += 1
    
    return vol

def normalize(vol):
    # stretch contrast to 0-255
    max_val = np.amax(vol)
    min_val = np.amin(vol)
    vol = (vol - min_val)/(max_val-min_val+1e-9)*255
    return np.uint8(vol)

def fix_borders(volume, conf_vol, block_size):
    volume = measure.label(volume, connectivity=1)
    # new_conf_vol = np.zeros(conf_vol.shape, conf_vol.dtype)
    # for i in np.delete(np.unique(volume), 0):
    #     conf_score = np.average(conf_vol[volume==i])
    #     new_conf_vol[volume==i] = conf_score
    # conf_vol = new_conf_vol
    # conf_vol[volume==0] = 0

    assert volume.shape == conf_vol.shape
    Z, H, W = volume.shape
    # volume: ZxHxW
    # merge vertically
    mapping = []
    for kk in tqdm(range(1,int(H/block_size))):
        im1 = volume[:, kk*block_size-1, :]
        im2 = volume[:, kk*block_size, :]
        # for all obj in im1, find if they have corresponding on im2
        im1_labels = np.unique(im1)
        im1_labels = im1_labels[im1_labels!=0]
        for l in im1_labels:
            z1,x1 = np.where(im1==l)
            y1 = np.full(z1.shape, kk*block_size-1)
            obj1_coords_set = set(tuple(zip(z1,x1)))

            im2_labels = np.unique(im2[z1,x1])
            im2_labels = im2_labels[im2_labels!=0]
            for ll in im2_labels:
                z2,x2 = np.where(im2==ll)
                y2 = np.full(z2.shape, kk*block_size)
                obj2_coords_set = set(tuple(zip(z2,x2)))
                if len(obj1_coords_set.intersection(obj2_coords_set))>10:
                    # mapping[(y2, x2, z2)] = (y1, x1, z1)
                    # mapping[(y2, x2, z2)] = l
                    mapping.append((l,ll))
        sub_volume = volume[:, kk*block_size-1-int(block_size/4):kk*block_size+1+int(block_size/4), :]
        sub_conf_vol = conf_vol[:, kk*block_size-1-int(block_size/4):kk*block_size+1+int(block_size/4), :]
        for k in mapping:
            mask0 = sub_volume==k[0]
            mask1 = sub_volume==k[1]
            sub_volume[mask1] = k[0]
            score1 = sub_conf_vol[mask0]
            score2 = sub_conf_vol[mask1]
            if len(score1)>0 and len(score2>0):
                new_socre = (score1[0]+score2[0])/2
            sub_conf_vol[mask0] = new_socre
            sub_conf_vol[mask1] = new_socre
    # for k,v in tqdm(mapping.items()):
        # fix confidence volume
        # assert len(np.unique(conf_vol[volume==k])) == 1
        # assert len(np.unique(conf_vol[volume==v])) == 1
        # # average the confidence score
        # avg_score = (np.average(np.unique(conf_vol[volume==k])) + np.average(np.unique(conf_vol[volume==v]))) / 2
        # conf_vol[volume==k] = avg_score
        # conf_vol[volume==v] = avg_score
        # fix volume labels
        # volume[volume==k] = v
        # volume[k[0], k[1], k[2]] = l
    # def myfun(a,b)
    # volume = np.vectorize(mapping.get)(volume)


    # volume = measure.label(volume, connectivity=1)
    # merge horizontally
    mapping = []
    for kk in tqdm(range(1,int(W/block_size))):
        im1 = volume[:, :, kk*block_size-1]
        im2 = volume[:, :, kk*block_size]
        # for all obj in im1, find if they have corresponding on im2
        im1_labels = np.unique(im1)
        im1_labels = im1_labels[im1_labels!=0]
        for l in im1_labels:
            z1,y1 = np.where(im1==l)
            obj1_coords_set = set(tuple(zip(z1,y1)))
            x1 = np.full(z1.shape, kk*block_size-1)
            im2_labels = np.unique(im2[z1,y1])
            im2_labels = im2_labels[im2_labels!=0]
            for ll in im2_labels:
                z2,y2 = np.where(im2==ll)
                x2 = np.full(z2.shape, kk*block_size)
                obj2_coords_set = set(tuple(zip(z2,y2)))
                if len(obj1_coords_set.intersection(obj2_coords_set))>10:
                    # mapping[ll] = l
                    mapping.append((l,ll))
        sub_volume = volume[:, :, kk*block_size-1-int(block_size/4):kk*block_size+1+int(block_size/4),]
        sub_conf_vol = conf_vol[:, :, kk*block_size-1-int(block_size/4):kk*block_size+1+int(block_size/4),]
        for k in mapping:
            mask0 = sub_volume==k[0]
            mask1 = sub_volume==k[1]
            sub_volume[mask1] = k[0]
            score1 = sub_conf_vol[mask0]
            score2 = sub_conf_vol[mask1]
            if len(score1)>0 and len(score2>0):
                new_socre = (score1[0]+score2[0])/2
            sub_conf_vol[mask0] = new_socre
            sub_conf_vol[mask1] = new_socre
    volume = measure.label(volume, connectivity=1)
    # for i in np.unique(volume):
    #     print(i, np.unique(conf_vol[volume==i]))
    return volume, conf_vol

def get_confidence_vol(seg, final_scores):
    # seg: 3d gray scale volume, nuclei are labeled with sequential intensity
    # final_scores: list, confidence score for each nuclei
    assert len(np.unique(seg))-1 == len(final_scores)
    conf = np.zeros(seg.shape, np.float32)
    labels = np.delete(np.unique(seg), 0)
    for label,score in zip(labels, final_scores):
        conf[seg==label] = score
    
    return conf

