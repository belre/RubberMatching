
import numpy as np

from PIL import Image
from sklearn.linear_model import Lasso, Ridge
from scipy.sparse import csr_matrix, lil_matrix

import cv2
import itertools
import os
import matplotlib.pyplot as plt
import glob

import json

basepath = '..\\..\image\\practice\\'

with open(basepath + 'image_reference.json', 'r') as f:
    image_region_info = json.load(f)

for info_item in image_region_info:
    if 'refer' in image_region_info[info_item] :
        filedir = basepath + image_region_info[info_item]['refer'] + ".bmp"
    else:
        filedir = basepath + info_item + ".bmp"

    resize_rate = image_region_info[info_item]['resize_rate']

    image = Image.open(filedir)
    #image_cv2 = cv2.imread(filedir)

    if 'roi' in image_region_info[info_item]:
        roi = image_region_info[info_item]['roi']
        image = image.crop((roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]))


    ### Background領域 - ヒストグラムの演算
    data_for_bg = np.array(image.getdata()).reshape(image.height, image.width)
    data_bg = np.array([])
    hist_bg = None
    if 'background_region' in image_region_info[info_item] :
        for bg_region in image_region_info[info_item]['background_region']:
            x = bg_region[0]
            y = bg_region[1]
            w = bg_region[2]
            h = bg_region[3]
            data_bg = np.append(data_bg, data_for_bg[y:y+h,x:x+w].flatten())

        hist_bg = np.histogram( 32 * np.log2(data_bg), range=(0,256), bins=256)
        avr_bg = np.average(32 * np.log2(data_bg))
        std_bg = np.std(32 * np.log2(data_bg))

    ### ここから
    image = image.resize(( int(image.width / resize_rate), int(image.height / resize_rate)))
    data = 32 * np.log2(np.array(image.getdata())[:]) #[t if t < 200 else 0 for t in image.getdata()])[:]
    size = image.size

    ### データを並べる
    data_1d = data.flatten()

    ### 対角行列を作る
    #H_diag = np.eye(data_1d.shape[0])
    H_csr = lil_matrix((data_1d.shape[0], data_1d.shape[0]))
    for i in range(0, data_1d.shape[0]):
        H_csr[i, i] = 1

    alpha_val = 0.0001
    if 'alpha' in image_region_info[info_item]:
        alpha_val = image_region_info[info_item]['alpha']

    model = Lasso(alpha=alpha_val).fit(H_csr, data_1d)
    image_raw_conv = model.coef_

    #image_raw_conv = np.power(2, image_raw_conv / 32)
    ### ここまで

    image_raw_conv = image_raw_conv.reshape(size[1], size[0])

    # 画像に変換
    image_raw_conv_8bit = image_raw_conv - np.min(image_raw_conv) + 50

    # 画像に対して何かの処理を追加する場合
    if 'attr' in image_region_info[info_item]:
        if 'background_adj' in image_region_info[info_item]['attr']:
            # 最頻値を求めて、閾値処理
            hist_lasso = np.histogram( image_raw_conv_8bit, range=(0,256), bins=256)
            hist_lasso_max = hist_lasso[1][np.argmax(hist_lasso[0])]
            image_raw_conv_8bit = np.where(image_raw_conv_8bit < hist_lasso_max, image_raw_conv_8bit, hist_lasso_max)
            pass

    image_conv_8bit = Image.fromarray(np.uint8(image_raw_conv_8bit))
    image_conv_8bit.save(basepath + '\\export\\' + os.path.basename(info_item) + '_.bmp')


    plt.figure(figsize=(17,9))
    plt.subplot(221)
    plt.title('Input Image')
    plt.imshow(image, cmap='gray')


    plt.subplot(222)
    plt.title('Filtered Image by L1 norm')
    plt.imshow(image_raw_conv, cmap='gray')

    plt.subplot(223)
    if hist_bg != None:
        plt.title('Histogram of Background')
        plt.bar(hist_bg[1][0:(len(hist_bg[0]))], hist_bg[0], width=1.0)

    #plt.subplot(234)
    #plt.imshow(data_reg, cmap='gray')

    plt.show()





pass

