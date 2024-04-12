###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
# Training code for EMR-CNN detectors
################################################################################################

import argparse
import os
from utils.config import Config
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from utils.dataset import get_nuclei_dicts
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
import time
# from options.train_options import options
setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, required=True,
                    help='name of the dataset')
parser.add_argument('--ensembleId', type=int, required=True,
                    help='id of the ensemble network')
parser.add_argument('--shuffle_models', type=bool, default=False,
                    help='if shuffle the backbone models')
parser.add_argument('--iters', type=int, default=1000,
                    help='number of iterations')
opt = parser.parse_args()   # get training options

config = Config(opt.data_name)
print("Initializaing Ensemble Network #" + str(opt.ensembleId))
# register dataset
for d in ["train"]:
    DatasetCatalog.register(
        config.data_name+"_" + d, lambda d=d: get_nuclei_dicts(config.train_img_dir, config.train_json_dir))

    MetadataCatalog.get(config.data_name+"_" + d).set(thing_classes=[config.data_name])

nuclei_metadata = MetadataCatalog.get(config.data_name+"_train")

start = time.time()
# Train dataset
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config.backbone_files[opt.ensembleId-1]))
print('loading backbbone ' + config.backbone_files[opt.ensembleId-1])
cfg.DATASETS.TRAIN = (config.data_name+"_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.backbone_files[opt.ensembleId-1])  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = opt.iters
# faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# cfg.INPUT.MIN_SIZE_TRAIN = 128
# cfg.INPUT.MAX_SIZE_TRAIN = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.OUTPUT_DIR = os.path.join(
    os.curdir, 'checkpoints', config.data_name, 'ensemble_'+str(opt.ensembleId))
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
end = time.time()
print('total training time for ensemble_'+str(opt.ensembleId) + ' is:', end - start)