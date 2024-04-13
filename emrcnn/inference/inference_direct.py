###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
# Code for directly inference EMR-CNN on a large volume without using divide-and-conquer strategy
################################################################################################

from utils.config import Config
import argparse
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from utils.CC2 import CC2
import math
import skimage.io as io
import os, cv2
import shutil
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import skimage.io as io
from skimage import measure
from utils.vis import apply_mask
from utils.inference import pred_volume_assemble
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', required=True, type=str, help='name of the dataset')
parser.add_argument('--save_vis', default=True, type=bool, help='if save the visualization images')
parser.add_argument('--save_masks', default=True, type=bool, help='if save the instance segmentation masks')
parser.add_argument('--save_scores', default=True, type=bool, help='if save the confidence scores')
parser.add_argument('--vis_bbox', default=False, type=bool, help='if draw the bounding box on the visualization image')
parser.add_argument('--vis_label', default=False, type=bool, help='if draw the label on the visualization image')

opt = parser.parse_args()   # get training options

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
for ensembleId in range(1, int(config.ensemble)+1):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.backbone_files[ensembleId-1]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = os.path.join(os.curdir, 'checkpoints', config.data_name, 'ensemble_'+str(ensembleId))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.DATASETS.TEST = (config.label_name + "_test", )
    predictors.append(DefaultPredictor(cfg))



# volume = volume[:32,:]
# window inference starts here
block=128
size_z, size_h, size_w, _ = volume.shape
w_p = int(math.ceil(size_w/float(block/2))*(block/2))
h_p = int(math.ceil(size_h/float(block/2))*(block/2))
z_p = int(math.ceil(size_z/float(block/2))*(block/2))

padz = z_p-size_z
padh = h_p-size_h
padw = w_p-size_w
print("x: " + str(h_p) + ", y: " + str(w_p) + ", z: "+ str(z_p))

input_numpy = volume.copy()
input_inference = input_numpy
output_sub, final_scores = pred_volume_assemble(config, predictors, input_numpy, block_size=input_numpy.shape[1])

fixed_gray, count = measure.label(output_sub, return_num=True, connectivity=1)
fixed_cc = CC2(fixed_gray, 0, 0)

overlay = apply_mask(volume[:,:,:,0], fixed_cc, 0.7)

if opt.save_masks:
    if os.path.exists(os.path.join(os.curdir, 'results', config.data_name, 'masks')):
        shutil.rmtree(os.path.join(os.curdir, 'results', config.data_name, 'masks/'))
    os.makedirs(os.path.join(os.curdir, 'results', config.data_name, 'masks'))
    io.imsave(os.path.join(os.curdir, 'results', config.data_name, 'masks', 'seg_vol.tif'), fixed_gray.astype(np.uint16))
    io.imsave(os.path.join(os.curdir, 'results', config.data_name, 'masks', 'seg_cc.tif'), fixed_cc)
    io.imsave(os.path.join(os.curdir, 'results', config.data_name, 'masks', 'overlay.tif'), overlay)
