###############################################################################################
# Copyright 2022 The Board of Trustees of Purdue University and the Purdue Research Foundation.
# All rights reserved.
# Implemented by Liming Wu
# Date: 01/20/2022
################################################################################################

from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage as ndi
from skimage import util
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist
from utils.config import Config
import os
import numpy as np
import cv2
import skimage.io as io
import time
from skimage import measure
from datetime import datetime
from pytz import timezone
import shutil
import math

def slice_merge_layercake(src_dir,dest_dir,radius=2, voxel_thresh=0):
    
    savepath = os.path.join(dest_dir, 'seg_results')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    vol_names = os.listdir(src_dir)
    vol_names.sort()
    for vv, vol_name in enumerate(vol_names):
        print('processing volume ', (vv+1))
        # for index, filename in enumerate(files_sorted):
        img_names = os.listdir(os.path.join(src_dir, vol_name))
        img_names.sort()
        cum_labels=0
        cents = []
        mapped = {}
        mapped[0]=0
        vol = None
        for index, img_name in enumerate(img_names):
            label = io.imread(os.path.join(src_dir,vol_name, img_name)).astype(np.uint16)
            shape = label.shape
            if vol is None:
                vol = np.zeros((len(img_names),shape[0],shape[1]),np.uint16)
            # assume the objects are labeled with unique incremental sequential number e.g. [0, 1, 2, 3, 4, 5, 6, 7]
            label = measure.label(label, connectivity=1).astype(np.uint16)
            max_label = np.max(label)
            
            label[label!=0] += cum_labels
            cum_labels += max_label
            vol[index,:,:] = label
            
            for ii in range(cum_labels+1-max_label,cum_labels+1):
                cents.append((index,)+center_of_mass(label==ii))
                mapped[ii]=ii

        #the actual layercake here. join if within some radius.
        # print(cents)
        #pairwise distance between centroids
        """dists = pdist(cents,lambda u, v: np.sqrt((u[2]-v[2])**2+(u[1]-v[1])**2))
        #The metric dist(u=X[i], v=X[j]) is computed and stored in entry ij
        for index,dist in enumerate(dists):
            if dist <= radius:
                (label_a,label_b) = condensed_to_square(index,cum_labels)
                if label_a > 0 and np.absolute(np.absolute(cents[label_b][0]-cents[label_a][0])-1)<0.5: #Check z component
                    mapped[label_b+1]=mapped[label_a+1]
        """
        for label_a in range(1,cum_labels+1):
            for label_b in range(label_a+1,cum_labels+1):
                diff = cents[label_b-1][0]-cents[label_a-1][0]
                if diff==1: #Check z component
                    u = cents[label_a-1]
                    v = cents[label_b-1]
                    if ((u[1]-v[1])**2+(u[2]-v[2])**2)<=radius*radius: # Check distance within radius
                        mapped[label_b]=mapped[label_a]
                if diff>1:
                    break
                    
        # print(mapped)
        # Apply mapped values to the volume
        vol=np.vectorize(mapped.get)(vol).astype(np.uint16)

        # small object removal
        cc = measure.label(vol,connectivity = 1)
        props = measure.regionprops(cc)
        for jj in range(0,np.amax(cc)):
            if props[jj].area < voxel_thresh:
                cc[props[jj].coords[:,0],props[jj].coords[:,1],props[jj].coords[:,2]] = 0
        vol = measure.label(cc,connectivity = 1).astype(np.uint16)


        #save to individual slices for visualization,
        if not os.path.exists(os.path.join(savepath, vol_name)):
            os.makedirs(os.path.join(savepath, vol_name))
        for zz in range(0,vol.shape[0]):
            cv2.imwrite(os.path.join(savepath, vol_name, img_names[zz]),vol[zz,:,:])
        savepath_3d = os.path.join(dest_dir,'seg_results_3d')
        if not os.path.exists(savepath_3d):
            os.makedirs(savepath_3d)
        io.imsave(os.path.join(savepath_3d,'vol_'+str(vv+1).zfill(2)+'.tif'),vol)
        
        #Color Coding
        ## Colormap - Read text files saved from MATLAB
        color_code_path = os.path.join(dest_dir, 'color_coded', vol_name)
        if not os.path.exists(color_code_path):
            os.makedirs(color_code_path)
        cmap = []
        thColormap = 50
        ins = open("utils/cmap.txt", "r")
        for line in ins:
            line = line.strip().split("\t")
            line2 = [float(n) for n in line]
            line3 = [int(line2[0]), int(line2[1]), int(line2[2])]
            cmap.append(line3)

        ins.close()

        cmap2 = []

        num_colors = 0
        # Dark color removal from colormap
        for i in range(0,len(cmap)):
            if cmap[i][0] > thColormap or cmap[i][1] > thColormap or cmap[i][2] > thColormap:
                cmap2.append(cmap[i])
                num_colors = num_colors + 1

        print("colormap done")
        Z = vol.shape[0]
        Y = vol.shape[1]
        X = vol.shape[2]
        bw3 = util.img_as_ubyte(np.zeros([Z,Y,X,3]))
        
        for ii in range(0,Z):	
            for jj in range(0,Y):
                for kk in range(0,X):
                    if vol[ii,jj,kk] != 0:
                        bw3[ii,jj,kk,:] = cmap2[(vol[ii,jj,kk]-1)%num_colors]
                        #bw3[ii,jj,kk,0] = cmap2[(vol[ii,jj,kk]-1)%num_colors][0]
                        #bw3[ii,jj,kk,1] = cmap2[(vol[ii,jj,kk]-1)%num_colors][1]
                        #bw3[ii,jj,kk,2] = cmap2[(vol[ii,jj,kk]-1)%num_colors][2]
            cv2.imwrite(os.path.join(color_code_path,"z%04d.png"%(ii+1)),bw3[ii,:,:,:])


#These functions are used for pdist
def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j
