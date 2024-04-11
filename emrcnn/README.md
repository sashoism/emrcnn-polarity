<!---
#Readme for EMRCNN Package 4/20/2022
#Copyright 2022 - The Board of Trustees of Purdue University - All rights reserved
#This is the training and testing code for EMRCNN.

#This software is covered by US patents and copyright.
#This source code is to be used for academic research purposes only, and no commercial use is allowed.
-->

*****
# Term of Use and License
Version 1.1
April 21, 2022


This work was partially supported by a George M. O’Brien Award from the National Institutes of Health 
under grant NIH/NIDDK P30 DK079312 and the endowment of the Charles William Harrison Distinguished 
Professorship at Purdue University.

Copyright and Intellectual Property

The software/code and data are distributed under Creative Commons license
Attribution-NonCommercial-ShareAlike - CC BY-NC-SA

You are free to:
* Share — copy and redistribute the material in any medium or format
* Adapt — remix, transform, and build upon the material
* The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
* Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. (See below for paper citation)
* NonCommercial — You may not use the material for commercial purposes.
* ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more information see:
https://creativecommons.org/licenses/by-nc-sa/4.0/


The data is distributed under Creative Commons license
Attribution-NonCommercial-NoDerivs - CC BY-NC-ND

You are free to:
* Share — copy and redistribute the material in any medium or format
* The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
* Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* NonCommercial — You may not use the material for commercial purposes.
* NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For More Information see:
https://creativecommons.org/licenses/by-nc-nd/4.0/



Attribution
In any publications you produce based on using our software/code or data we ask that you cite the following paper:
```BibTeX
@ARTICLE{wu2022emrcnn,
  author={Liming Wu, Alain Chen, Paul Salama, Kenneth W. Dunn, and Edward J. Delp},
  title={An Ensemble Learning and Slice Fusion Strategy for Three-Dimensional Nuclei Instance Segmentation},
  Journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2022},
  month={June}
}
```

Privacy Statement
We are committed to the protection of personal privacy and have adopted a policy to  protect information about individuals. 
When using our software/source code we do not collect any information about you or your use of the code.

How to Contact Us

The source code/software and data is distributed "as is." 
We do not accept any responsibility for the operation of the software.
We can only provided limited help in running our code. 
Note: the software/code developed under Linux. 
We do not do any work with the Windows OS and cannot provide much help in running our software/code on the Windows OS

If you have any questions contact: imart@ecn.purdue.edu

*****

# EMRCNN Training and Inference
This repository contains **training** and **inference** code for the EMRCNN.

## Installation
This project is built using [PyTorch](https://pytorch.org), a high-level library for Deep Learning. We tested the code under the environment of Ubuntu 20.04 with CUDA-11.0. 

Note: This package has been developed and compiled to run on a Linux machine and will not work with on any other machine that is running a different OS such Windows OS.
 

### Software installation requirements:
1. to use this project, at least one TITAN Xp GPU with 12GB memory is reguired
2. use the command `conda env create -f environment.yml` to create a virtual environment with required packages
3. install [PyTorch](https://pytorch.org/get-started/locally/) based on your CUDA version
4. install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) based on your CUDA version and PyTorch version

Note that if error occurs while running training or testing code, please check if Detectron2 version is compatable with PyTorch and CUDA version.

## Repository Structure 
This repository is structured in the following way: 
* `scripts/`: bash files for training, testing, and inference
* `checkpoints/`: pretrained model parameters
* `dataset/`: dataset used for EMRCNN, an example dataset is provided
* `result/`:example images generated when inferecing the provided pre-trained models
* `environment.yml`: virtual environment packages needed to install
* `utils/`: contains weighted-masks-fusion and slice merging modules and other utility functions
* `analysis_scripts/`: code for robustness and running time analysis described in the paper
* `inference/`: scripts used for inference on large volumes using divide-and-conquer strategy

## Dataset
One example dataset is provided as in `dataset/immu`.
Before training:
1. unzip it from the `dataset/` folder using command `unzip dataset/immu.zip -d dataset/`
2. refer to the README.md file in `dataset/immu` for more details of the folder structure
3. run `python dataset/createPolygons.py` to generate JSON files as ground truth

## How to use the code

### Training
The example training data is provided in `dataset/immu/train/`, these are synthetic microscopy images generated from SpCycleGAN. The source code of SpCycleGAN is also provided. To train on example data:
1. set your root path of this project for variable `project_dir` in file `utils/config.py`
2. make appropriate modification in file `scripts/train_ensemble.sh` based on your GPUs
3. run command `sh scripts/train_ensemble.sh`, the weights will be saved in `checkpoints/immu_ensemble`

### Training on your own data:

If you are using your own data, please mimic the **immu** data settings in `utils/config.py` and directory structures in `dataset/immu`
1. orgnize your own data to the same structure as the example **immu** data under `dataset/immu/`
2. in file `utils/config.py`, add another `if` statement with same variables of `immu` and add these pathes and data information
3. make appropriate modification in file `scripts/train_ensemble.sh` based on your GPUs
4. run command `sh scripts/train_ensemble.sh`, the weights will be saved in `checkpoints/immu_ensemble`

### Using trained models

If your testing volume is the same size as the training volume, you can do **testing**:
1. set your root path of this project for variable `project_dir` in file `utils/config.py`
2. make sure other settings are correct in file `utils/config.py`
3. make appropriate changes in `scripts/test_ensemble.sh` 
4. run command `sh scripts/test_ensemble.sh`, this will save detected results on each slice and save to `results/`
5. to fuse results from all detectors, make appropriate file name changes in `ensemble_fusion.py` and run `python ensemble_fusion.py`
6. to fuse 2D segmentation results to 3D segmentation results, run `python slice_merge.py`
7. find the results under `results/immu_ensemble/weighted_mask_fusion/AHC/mask_overlay/`

Note that if you want to reproduce the results shown in the paper, you need to use 8 different detectors instead of 4. Also, you need to resize all images under `dataset/immu/train/` and `dataset/immu/test1/` from 128x128 to 512x512 and then run `dataset/createPolygons.py`. After training the EMR-CNN, you need to resize results back to 128x128 for evaluation. 
We now train and test on 128x128 to save some time while inference on large microscopy volumes.

If your testing volume is very large, you can do **inference**, which uses divide-and-conquer strategy:
1. modify the settings in `inference/inference_divide-and-conquer.py`
2. run command `python inference_divide-and-conquer.py`

## References
The Mask RCNN detector used in this paper is from [Detectron2 Modell Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md), for detail of the implementation please check out the original [Detectron2](https://github.com/facebookresearch/detectron2) code if you would like to understand more and reproduce the results. Detectron2 is released under the [Apache 2.0 license](http://www.apache.org/licenses/).