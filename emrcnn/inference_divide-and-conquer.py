###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from utils.config import Config
import argparse
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import time
import math
import skimage.io as io
import os, cv2
from utils.CC2 import CC2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import skimage.io as io
from utils.util_funcs import fix_borders
from utils.inference import pred_volume_assemble
from utils.vis import apply_mask
import warnings
warnings.filterwarnings("ignore")

print("The PID of this process is: ", os.getpid())
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', type=str, help='root dir of this project, needs to be modified')
parser.add_argument('--data_name', type=str, help='name of the dataset')
parser.add_argument('--model_dir', type=str, help='directory of checkpoints')
parser.add_argument('--test_img_dir', type=str, help='test image director containing all Z-slices of volume')
parser.add_argument('--M', type=int, default=4, help='number of models in an ensemble')
parser.add_argument('--block_size', type=int, default=128, help='size of the inference block')
parser.add_argument('--margin_size', type=int, default=16, help='size of the padded margin')
parser.add_argument('--outlier_thresh', type=int, default=1, help='number of elements in a cluster less than which is considered as outlier')
parser.add_argument('--voxel_thresh', type=int, default=200, help='number of voxels less than which the obj will be revmoed during color coding')
parser.add_argument('--k_min', type=int, default=3, help='minimum possible number of nuclei in an inference patch (Z x block_size x block_size)')
parser.add_argument('--k_max', type=int, help='maximum possible number of nuclei in an inference patch (Z x block_size x block_size)')
opt = parser.parse_args()   # get training options

opt.project_dir = "/data/wu1114/Documents/emrcnn_release/"
opt.data_name = "immu_large_ensemble"
opt.M = 4
opt.model_dir= "immu_ensemble"
opt.test_img_dir = os.path.join(opt.project_dir, "dataset/immu/test2/real")
opt.k_max = 100

config = Config(opt.data_name)
# read the entire volume
dataset_dicts = sorted(os.listdir(opt.test_img_dir))
volume = None
for i, d in enumerate(dataset_dicts):
    im = cv2.imread(os.path.join(opt.test_img_dir, d))
    if volume is None:
        volume = np.zeros(((len(dataset_dicts),) + im.shape), np.uint8)
    volume[i, :, :, :] = im

# load all models
predictors = []
for ensembleId in range(1, opt.M+1):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.backbone_files[ensembleId-1]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = os.path.join(os.curdir, 'checkpoints', opt.model_dir, 'ensemble_'+str(ensembleId))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    # cfg.DATASETS.TEST = (opt.label_name + "_test", )
    predictors.append(DefaultPredictor(cfg))

# window inference starts here
block=opt.block_size
margin=opt.margin_size
size_z, size_h, size_w, _ = volume.shape
w_p = int(math.ceil(size_w/float(block))*(block))
h_p = int(math.ceil(size_h/float(block))*(block))
z_p = size_z

padz = z_p-size_z
padh = h_p-size_h
padw = w_p-size_w
print("inference block size: "+str(block))
print("original volume size: ZxYxX = ", volume.shape[:-1])

input_numpy = volume.copy()
input_inference = np.pad(input_numpy, ((0, padz), (0, padh), (0, padw), (0, 0)), 'constant') # pad to make it okay to inference
input_inference = np.pad(input_inference, ((0, 0), (margin, margin), (margin, margin), (0, 0)), 'constant') # pad margins
print("padded volume size: ZxYxX = ", input_inference.shape[:-1])

output_gray = np.zeros([z_p,h_p,w_p])  # 1 channel
output_conf = np.zeros([z_p,h_p,w_p])  # confidence map
output_cc = np.zeros([z_p,h_p,w_p,3])  # 3 channel
for kk in range(0,int(w_p/block)):
    for jj in range(0,int(h_p/block)):
            print(block*jj,':',block*(jj+1),' ', block*kk,':',block*(kk+1))
            inputs_sub = input_inference[:,block*jj:(block*(jj+1)+margin*2),block*kk:(block*(kk+1)+margin*2),:]
            output_sub, conf_sub = pred_volume_assemble(opt, predictors, inputs_sub, block+2*margin)
            output_conf[:,block*jj:(block*(jj+1)),block*kk:(block*(kk+1))] = conf_sub[:,margin:block+margin, margin:block+margin]
            output_gray[:,block*jj:(block*(jj+1)),block*kk:(block*(kk+1))] = output_sub[:,margin:block+margin, margin:block+margin]

output_gray = output_gray[0:size_z,0:size_h,0:size_w]
output_conf = output_conf[0:size_z,0:size_h,0:size_w]
# io.imsave('./inference/'+opt.data_name+'_output_gray2.tif', output_gray.astype(np.uint8))
# io.imsave('./inference/'+opt.data_name+'_output_conf2.tif', output_conf)
# output_gray = io.imread('./inference/'+opt.data_name+'_output_gray.tif')
# output_conf = io.imread('./inference/'+opt.data_name+'_output_conf.tif')
# divide-and-conquer fix the boundary nuclei
fixed_gray, conf_vol = fix_borders(output_gray, output_conf, block)
fixed_gray, conf_vol = fix_borders(fixed_gray, conf_vol, block)

cc = CC2(fixed_gray, 0, 0)

overlay = apply_mask(volume[:,:,:,0], cc, 0.7)

# save results
session_name = 'Inference_session' + '_' + time.strftime('%m.%d_%Hh%M')
os.makedirs(os.path.join(os.curdir, 'results', opt.data_name, session_name))
io.imsave(os.path.join(os.curdir, 'results', opt.data_name, session_name, 'seg_vol.tif'), fixed_gray.astype(np.uint16))
io.imsave(os.path.join(os.curdir, 'results', opt.data_name, session_name, 'seg_cc.tif'), cc)
io.imsave(os.path.join(os.curdir, 'results', opt.data_name, session_name, 'overlay.tif'), overlay)
