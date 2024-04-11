###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
# code for converting ground truth masks to JSON files read by Detectron2
################################################################################################

import numpy as np
from skimage import data, img_as_float
from skimage.util import invert
import skimage.draw
import skimage.io as io
import os
import json
from tqdm import tqdm
from imantics import Polygons, Mask

def encode_images():
    # encode images as json file for training Detectron2
    # follows the json format of https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon
    path = 'dataset/immu/train'
    dest = 'dataset/immu/train/json'
    gt_img_names = os.listdir(os.path.join(path, 'gt'))
    gt_img_names.sort()
    data = {}
    for gt_img_name in tqdm(gt_img_names):
        gt = io.imread(os.path.join(path,'gt', gt_img_name))       
        # encode the image to json file
        file_size = str(os.path.getsize(os.path.join(path, 'syn')))
        id_ = gt_img_name + file_size
        nuclei_intensity = list(np.unique(gt))
        nuclei_intensity.remove(0)
        obj = {}
        idx = 0
        for i in nuclei_intensity:
            gt_tmp = np.uint8(gt==i)*255
            # get polygons
            polygons = Mask(gt_tmp).polygons()
            all_points_x = polygons.points[0][:,0].tolist()  # should be a list
            all_points_y = polygons.points[0][:,1].tolist()
            if len(all_points_x) < 6:
                continue
            if len(all_points_x)%2 != 0:
                all_points_y.append(all_points_y[-1])
                all_points_x.append(all_points_x[-1])
            obj[idx] = {'shape_attributes':{'name':'polygon', 'all_points_x':all_points_x, 'all_points_y':all_points_y}, 
            'region_attributes':{}}
            idx += 1
        data[id_] = {'fileref':'', 'size':file_size, 'filename':gt_img_name, 'base64_img_data':'', 'file_attributes':{}, 'regions':obj}
    if not os.path.exists(dest):
        os.makedirs(dest)
    with open(os.path.join(dest, 'via_region_data.json'), 'w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    encode_images()
