
import numpy as np

from PIL import Image
from sklearn.linear_model import Lasso, Ridge
from scipy.sparse import csr_matrix, lil_matrix

import cv2
import itertools
import os

import matplotlib.pyplot as plt
import matplotlib.pyplot as patches


import glob

import json

basepath = '..\\..\image\\practice\\'

with open(basepath + 'pattern_matching.json', 'r') as f:
    tp_info = json.load(f)

### テンプレート画像を読込
image_tp = cv2.imread(basepath + tp_info['reference']['filepath'])
image_view_tp = cv2.imread(basepath + tp_info['reference']['imgpath'])
roi_list = tp_info['reference']['roi']

# roi保存
for name, roi in roi_list.items():
    image_tp_roi = image_view_tp[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.imwrite(basepath + tp_info['reference']['imgpath'] + "_" + name + ".bmp", image_tp_roi)
    pass


### サンプル画像
for item in tp_info['sample']:

    # サンプル画像読込
    image_sample = cv2.imread(basepath + item + '.bmp')
    image_sample = cv2.GaussianBlur( image_sample, (1,1), 0)

    result = cv2.matchTemplate( image_sample, image_tp, cv2.TM_CCOEFF)
    loc_result = cv2.minMaxLoc(result)

    plt.figure(figsize=(17,9))

    ax = plt.subplot(131)
    plt.title('Template Image')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(image_tp, cmap='gray')
    for name, roi in roi_list.items():
        ax.add_patch(patches.Rectangle(xy=(roi[0], roi[1]), width=roi[2], height=roi[3], ec='g', fill=False))
        
    
    ax = plt.subplot(132)
    plt.title('Template Image(VIEW)')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(image_view_tp, cmap='gray')
    for name, roi in roi_list.items():
        ax.add_patch(patches.Rectangle(xy=(roi[0], roi[1]), width=roi[2], height=roi[3], ec='g', fill=False))
        
    
    ax = plt.subplot(133)
    plt.title('Input Image (After pre-positioning)')
    plt.imshow(image_sample[loc_result[3][1]:loc_result[3][1]+image_tp.shape[0],loc_result[3][0]:loc_result[3][0]+image_tp.shape[1]] , cmap='gray')    
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    for name, roi in roi_list.items():
        ax.add_patch(patches.Rectangle(xy=(roi[0], roi[1]), width=roi[2], height=roi[3], ec='g', fill=False))
        

    plt.show()
