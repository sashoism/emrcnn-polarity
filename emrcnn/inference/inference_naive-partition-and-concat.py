###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from numpy.testing import measure
from utils.config import Config
import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import sys
import time
import math
import skimage.io as io
import os, json, cv2, random
import shutil
from utils.CC2 import CC2
from utils.util_funcs import relabel_sequentially
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
import skimage.io as io
import skimage.measure as measure
from utils.inference import pred_volume_assemble
from utils.results_organize import imgs2imgs
from utils.vis import apply_mask
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, required=True, help='name of the dataset')
parser.add_argument('--model_dir', type=str, required=True, help='directory of checkpoints')
# parser.add_argument('--ensembleId', type=int, required=True, help='id of the ensemble network')
parser.add_argument('--label_name', type=str, required=True, help='label shown on the detection mask')
parser.add_argument('--save_vis', default=True, type=bool, help='if save the visualization images')
parser.add_argument('--save_masks', default=True, type=bool, help='if save the instance segmentation masks')
parser.add_argument('--save_scores', default=True, type=bool, help='if save the confidence scores')
parser.add_argument('--vis_bbox', default=False, type=bool, help='if draw the bounding box on the visualization image')
parser.add_argument('--vis_label', default=False, type=bool, help='if draw the label on the visualization image')

opt = parser.parse_args()   # get training options
# opt.data_name = "intestine_large_ensemble"
# opt.ensembleId=4
# opt.label_name="intestine"
# opt.save_vis = True
# opt.save_vis = True
# opt.save_masks = True
# opt.save_scores = True
# opt.vis_bbox = False
# opt.vis_label = False

config = Config(opt.data_name)
# read the entire volume
dataset_dicts = sorted(os.listdir(config.test_img_dir))
volume = np.zeros((32, 512, 512, 3), np.uint8)
for i, d in enumerate(dataset_dicts):
    im = cv2.imread(os.path.join(config.test_img_dir, d))
    volume[i, :, :, :] = im

# volume = np.pad(volume, ((32,32), (32,32), (32,32), (0,0)), mode='constant')
# load all models
predictors = []
for ensembleId in range(1, 5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.backbone_files[ensembleId-1]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = os.path.join(os.curdir, 'checkpoints', opt.model_dir, 'ensemble_'+str(ensembleId))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.DATASETS.TEST = (opt.label_name + "_test", )
    predictors.append(DefaultPredictor(cfg))



# volume = volume[:32,:]
# window inference starts here
block=128
size_z, size_h, size_w, _ = volume.shape
w_p = int(math.ceil(size_w/float(block))*(block))#512
h_p = int(math.ceil(size_h/float(block))*(block))#512
z_p = size_z

padz = z_p-size_z
padh = h_p-size_h
padw = w_p-size_w
print("x: " + str(h_p) + ", y: " + str(w_p) + ", z: "+ str(z_p))

input_numpy = volume.copy()
input_inference = np.pad(input_numpy, ((0, padz), (0, padh), (0, padw), (0, 0)), 'constant')
output_gray = np.zeros([z_p,h_p,w_p])  # 1 channel
output_cc = np.zeros([z_p,h_p,w_p,3])  # 3 channel
for kk in range(0,int(w_p/block)):
    for jj in range(0,int(h_p/block)):
            print(block*jj,':',block*(jj+1),block*kk,':',block*(kk+1))
            inputs_sub = input_inference[:,block*jj:block*(jj+1),block*kk:block*(kk+1),:]
            output_sub, final_scores = pred_volume_assemble(config, predictors, inputs_sub, block)
            output_gray[:,block*jj:block*(jj+1),block*kk:block*(kk+1)] = output_sub

output_gray = output_gray[0:size_z,0:size_h,0:size_w]
output_cc = output_cc[0:size_z,0:size_h,0:size_w,:]

# divide-and-conquer fix the boundary nuclei
fixed_gray, count = measure.label(output_gray, return_num=True, connectivity=1)
fixed_cc = CC2(fixed_gray, 0, 0)

overlay = apply_mask(volume[:,:,:,0], fixed_cc, 0.7)

if opt.save_masks:
    if os.path.exists(os.path.join(os.curdir, 'results', opt.data_name, 'masks')):
        shutil.rmtree(os.path.join(os.curdir, 'results', opt.data_name, 'masks/'))
    os.makedirs(os.path.join(os.curdir, 'results', opt.data_name, 'masks'))
    io.imsave(os.path.join(os.curdir, 'results', opt.data_name, 'masks', 'seg_vol.tif'), fixed_gray.astype(np.uint16))
    io.imsave(os.path.join(os.curdir, 'results', opt.data_name, 'masks', 'seg_cc.tif'), fixed_cc)
    io.imsave(os.path.join(os.curdir, 'results', opt.data_name, 'masks', 'overlay.tif'), overlay)
