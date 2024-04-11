###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plot
from tqdm import tqdm
import skimage.io as io
from skimage.color import rgb2gray
from skimage import measure
from scipy.ndimage.measurements import label
from utils.vis import apply_mask

def mask_overlay(img_path, root_dir, img_size, alpha):
    """[overlay mask volumes to the original volumes]

    Args:
        img_path ([type]): [description]
        color_code_path ([type]): [description]
        alpha ([type]): [description]
    """
    cc_dir = os.path.join(root_dir, 'color_coded')
    dest_dir = os.path.join(root_dir, 'mask_overlay')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if isinstance(img_size, tuple):
        img_size = img_size
    elif isinstance(img_size, int):
        img_size = (img_size, img_size)
    # read volumes
    segnames = sorted(os.listdir(cc_dir))
    imgnames = sorted(os.listdir(img_path))
    for i, (imgname, segname) in enumerate(zip(imgnames, segnames)):
        segfiles = sorted(os.listdir(os.path.join(cc_dir, segname)))
        imgfiles  = sorted(os.listdir(os.path.join(img_path, imgname)))
        # reading img slices
        img_vol = np.zeros((len(imgfiles), img_size[0], img_size[1]), dtype=np.uint8)
        seg_vol = np.zeros((len(imgfiles), img_size[0], img_size[1], 3), dtype=np.uint8)
        for z, (imgfile, segfile) in enumerate(zip(imgfiles, segfiles)):
            img_vol[z] = io.imread(os.path.join(img_path, imgname, imgfile))
            seg_vol[z] = io.imread(os.path.join(cc_dir, segname, segfile))
        overlay_vol = apply_mask(img_vol.copy(), seg_vol, alpha=alpha)
        
        # save overlayed volumes
        io.imsave(os.path.join(dest_dir, segname+'.tif'), overlay_vol)

