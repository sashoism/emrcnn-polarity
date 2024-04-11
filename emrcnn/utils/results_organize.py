###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

import os
import numpy as np
import skimage.io as io
import shutil

def imgs2imgs(src, v, z):
    # v: number of volumes
    # z: number of slices for each volume
    file_names = os.listdir(src)
    file_names.sort()
    if isinstance(z, list):
        for i in range(0, v):
            if not os.path.exists(os.path.join(src, 'vol'+str(i+1).zfill(3))):
                os.makedirs(os.path.join(src, 'vol'+str(i+1).zfill(3)))
            batch_names = file_names[0:z[i]]
            for name in batch_names:
                shutil.move(os.path.join(src, name), os.path.join(src, 'vol'+str(i+1).zfill(3)))
            del file_names[0:z[i]]
    else:
        for i in range(0, v):
            if not os.path.exists(os.path.join(src, 'vol'+str(i+1).zfill(3))):
                os.makedirs(os.path.join(src, 'vol'+str(i+1).zfill(3)))
            batch_names = file_names[i*z:(i+1)*z]
            for name in batch_names:
                shutil.move(os.path.join(src, name), os.path.join(src, 'vol'+str(i+1).zfill(3)))

