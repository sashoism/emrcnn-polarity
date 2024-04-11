###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from utils.dataset import get_nuclei_dicts
import numpy as np
import skimage.io as io
import os
import json
import cv2
import random
from skimage import measure
from skimage.color import rgb2gray
from skimage.transform import rescale
import matplotlib.pyplot as plt
import torch
from utils.util_funcs import split_intensities, normalize
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode

# check dataset
def plot_sample(img_dir, json_dir):
    dataset_dicts = get_nuclei_dicts(img_dir, json_dir)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=nuclei_metadata, scale=2)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image()[:, :, ::-1])
        io.imsave('demo.png', out.get_image()[:, :, ::-1])
        break

def draw_cluster(prob_mask, centers, labels):
    color_rgb = np.load('utils/color_rgb.npy')
    prob_mask = np.uint8(prob_mask/np.max(prob_mask)*255)
    rgb_mask = cv2.cvtColor(prob_mask, cv2.COLOR_GRAY2RGB)
    for i in np.unique(labels):
        cluster_centers = centers[labels==i]
        c = tuple(color_rgb[i % len(color_rgb)])
        for cluster_center in cluster_centers:
            cv2.circle(rgb_mask, (int(cluster_center[1]), int(cluster_center[0])), radius=2,
                       color=(int(c[0]), int(c[1]), int(c[2])), thickness=-1)
    
    return rgb_mask



def draw_on_img(orig_img, masks, scores, opt):
    # given an image and its corresponding masks and scores
    # draw the mask and scores on the image
    orig_img = rescale(orig_img, 0.25, order=1, preserve_range=True)
    for d in ["test"]:
        MetadataCatalog.get(opt.label_name+"_" + d).set(thing_classes=[opt.label_name])
    nuclei_metadata = MetadataCatalog.get(opt.label_name+"_test")
    v = Visualizer(orig_img,
                   metadata=nuclei_metadata,
                   scale=4,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW)
    masks_split = split_intensities(masks)
    # create Instance
    ins = Instances(orig_img.shape, scores=torch.tensor(scores), pred_masks=torch.tensor(masks_split))
    if scores != None:
        out=v.draw_instance_predictions(ins)
    else:
        # Do not draw labels
        outputs_tmp=ins.copy()
        outputs_tmp.remove("scores")
        outputs_tmp.remove("pred_classes")
        out = v.draw_instance_predictions(outputs_tmp)
    # io.imsave(os.path.join(os.curdir, 'results', opt.experiment_name, 'ensemble_'+str(opt.ensembleId),
                            # 'visualize', d.split('/')[-1]), out.get_image()[:, :, ::-1]))
    return out.get_image()[:, :, ::-1]


'''
Utility function for overlay color coded masks to the orignal microscopy volumes
'''
def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image (3d)

    Args:
        image ([ZxXxYx3 numpy array]): overlay mask on image
        mask ([ZxXxYx3 numpy array]): color coded mask
        alpha (float, optional): [description]. Defaults to 0.5.

    Returns:
        [ZxXxYx3]: [overlayed volume]
    """
    # color mask to gray-scale
    image = np.stack((image,)*3, axis=-1)
    gray_mask = np.uint8(rgb2gray(mask)*255)
    cc = measure.label(gray_mask, connectivity = 1)
    props = measure.regionprops(cc)
    # for ii in tqdm(range(1,np.amax(cc)+1)):
    for c in range(3):
        image[:, :, :, c] = np.where(cc > 0, image[:, :, :, c] * (1 - alpha) + alpha * mask[:,:,:,c],
                                image[:, :, :, c])
    return image

def visualize_confidence(conf_vol, intervals=[0, 0.7, 0.8, 0.9, 1.0]):
    conf_vol_cc = np.zeros(conf_vol.shape, conf_vol.dtype)

    for i in range(len(intervals)-1):
        low, high = intervals[i], intervals[i+1]
        conf_vol_cc[np.logical_and(conf_vol>low, conf_vol<high)] = i

    conf_vol_cc = normalize(conf_vol_cc)
    # save confidence score map (color coded)
    conf_vol_cc2 = np.zeros(conf_vol_cc.shape+(3,), dtype=np.uint8)
    for z in range(conf_vol_cc.shape[0]):
        conf_vol_cc2[z] = cv2.applyColorMap(conf_vol_cc[z], cv2.COLORMAP_JET)

    return conf_vol_cc2, conf_vol_cc
        