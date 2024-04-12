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
import skimage.io as io
import os, cv2
import shutil
from utils.util_funcs import relabel_sequentially
from skimage.transform import rescale, resize
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from utils.results_organize import imgs2imgs

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, help='name of the dataset')
parser.add_argument('--ensembleId', type=int, help='id of the ensemble network')
parser.add_argument('--save_vis', default=True, type=bool, help='if save the visualization images')
parser.add_argument('--save_masks', default=True, type=bool, help='if save the instance segmentation masks')
parser.add_argument('--save_scores', default=True, type=bool, help='if save the confidence scores')
parser.add_argument('--vis_bbox', default=False, type=bool, help='if draw the bounding box on the visualization image')
parser.add_argument('--vis_label', default=False, type=bool, help='if draw the label on the visualization image')

opt = parser.parse_args()   # get training options

config = Config(opt.data_name)
# register dataset
for d in ["test"]:
    MetadataCatalog.get(config.label_name+"_" + d).set(thing_classes=[config.label_name])
nuclei_metadata = MetadataCatalog.get(config.label_name+"_test")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config.backbone_files[opt.ensembleId-1]))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.OUTPUT_DIR = os.path.join(os.curdir, 'checkpoints', opt.data_name, 'ensemble_'+str(opt.ensembleId))

# test model on valication set
# cfg.INPUT.MIN_SIZE_TEST = 128
# cfg.INPUT.MAX_SIZE_TEST = 128

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
cfg.DATASETS.TEST = (config.label_name + "_test", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
import skimage.io as io
dataset_dicts = sorted(os.listdir(config.test_img_dir))

if opt.save_masks:
    if os.path.exists(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId),'masks')):
        shutil.rmtree(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'masks/'))
    os.makedirs(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'masks'))
if opt.save_vis:
    if os.path.exists(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'visualize')):
        shutil.rmtree(os.path.join(os.curdir, 'results', opt.data_name,'ensemble_'+str(opt.ensembleId), 'visualize/'))
    os.makedirs(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'visualize'))
if opt.save_scores:
    if os.path.exists(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'scores')):
        shutil.rmtree(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId), 'scores/'))
    os.makedirs(os.path.join(os.curdir, 'results', opt.data_name,'ensemble_'+str(opt.ensembleId), 'scores'))
start = time.time()
for i, d in enumerate(dataset_dicts):
    # if i == 54:
    #     print()
    im = cv2.imread(os.path.join(config.test_img_dir, d))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=nuclei_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    if opt.save_vis:
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        outputs_tmp = outputs["instances"].to("cpu")
        if not opt.vis_label:
            outputs_tmp.remove("scores")
            outputs_tmp.remove("pred_classes")
        if not opt.vis_bbox:
            outputs_tmp.remove("pred_boxes")
        out = v.draw_instance_predictions(outputs_tmp)
        io.imsave(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_'+str(opt.ensembleId),
                            'visualize', d.split('/')[-1]), out.get_image()[:, :, ::-1])

    # Save instance masks
    if isinstance(config.orig_img_size, tuple):
        im_size = config.orig_img_size
    elif isinstance(config.orig_img_size, int):
        im_size = (config.orig_img_size, config.orig_img_size)

    removed_obj_idx = []    # obj are needs to be removed due to overlapping
    if opt.save_masks:
        masks = outputs["instances"].to("cpu").pred_masks
        mask_combined = np.zeros(im_size, dtype=np.uint16)
        for ii in range(0, masks.shape[0]):
            mask = np.uint8(masks[ii,:,:])*255
            mask = rescale(mask, (im_size[0]/mask.shape[0], im_size[1]/mask.shape[1]), order=1)
            # to solve the problem that next object overlay the former object
            if np.all(np.logical_and(mask>0.5, mask_combined==0)==False):
                removed_obj_idx.append(ii)
            mask_combined[np.logical_and(mask>0.5, mask_combined==0)] = ii+1
        if len(removed_obj_idx)!=0:
            mask_combined = relabel_sequentially(mask_combined)
        io.imsave(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_' +
                            str(opt.ensembleId), 'masks', d.split('/')[-1]), mask_combined)
    
    # also need to save probability scores used for weighted pixel fusion
    if opt.save_scores:
        scores = np.delete(outputs["instances"].to("cpu").scores, removed_obj_idx)
        np.save(os.path.join(os.curdir, 'results', opt.data_name, 'ensemble_' + str(opt.ensembleId), 'scores', d.split('.')[0]+'.npy'), scores)

if opt.save_masks:
    imgs2imgs(src=os.path.join('results', opt.data_name, 'ensemble_' + str(opt.ensembleId), 'masks'), v=config.v, z=config.z)
if opt.save_scores:
    imgs2imgs(src=os.path.join('results', opt.data_name, 'ensemble_' + str(opt.ensembleId), 'scores'), v=config.v, z=config.z)
if opt.save_vis:
    imgs2imgs(src=os.path.join('results', opt.data_name, 'ensemble_' + str(opt.ensembleId), 'visualize'), v=config.v, z=config.z)

# Save segmentation masks for layercake, save centroid information
print('total testing time for ensemble_'+str(opt.ensembleId) + ' is:', (time.time() - start))