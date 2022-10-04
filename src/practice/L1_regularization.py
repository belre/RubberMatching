
import numpy as np

from PIL import Image
from sklearn.linear_model import Lasso, Ridge

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
    image_cv2 = cv2.imread(filedir)

    ### Background領域 - ヒストグラムの演算
    data_for_bg = np.array(image.getdata()).reshape(image.height, image.width)

    data_bg = np.array([])
    for bg_region in image_region_info[info_item]['background_region']:
        x = bg_region[0]
        y = bg_region[1]
        w = bg_region[2]
        h = bg_region[3]
        data_bg = np.append(data_bg, data_for_bg[y:y+h,x:x+w].flatten())

    hist_bg = np.histogram( 20 * np.log2(data_bg), range=(0,160), bins=161)
    avr_bg = np.average(20 * np.log2(data_bg))
    std_bg = np.std(20 * np.log2(data_bg))

    print(str(avr_bg) + "," + str(std_bg))


    """
    ### Background領域をµ,σを使って正規化する
    pix_margin = 40
    data = np.array(image.getdata())[:] #.reshape(image.widht, image.height)
    data = 20 * np.log2(data.reshape(image.height, image.width))
    data_reg = np.zeros((data.shape[0], data.shape[1]))
    for k in range(0, data.shape[0]):
        for l in range(0, data.shape[1]):           
            k_s = int(k - pix_margin / 2 if k >= pix_margin else 0)
            k_e = int(k + pix_margin / 2 if k - pix_margin < data.shape[0] else data.shape[0]-1)
            l_s = int(l - pix_margin / 2 if l >= pix_margin else 0)
            l_e = int(l + pix_margin / 2 if l - pix_margin < data.shape[1] else data.shape[1]-1)
            
            data_trim = data[k_s:k_e, l_s:l_e]
            avr_trim = np.average(data_trim)
            std_trim = np.std(data_trim)

            data_reg[k, l] = avr_trim - avr_bg

            
            if np.abs(avr_trim - avr_bg) < 10 and std_trim / std_bg >= 0.5 and std_trim / std_bg < 2:
                data_reg[k, l] = 0
            else: 
                data_reg[k, l] = data[k, l]
            pass

    image_reg = Image.fromarray(data_reg)
    """

    #image_edge_cv2 = cv2.Canny(image_cv2, 20, 50)

    image = image.resize(( int(image.width / resize_rate), int(image.height / resize_rate)))
    data = 20 * np.log2(np.array(image.getdata())[:]) #[t if t < 200 else 0 for t in image.getdata()])[:]
    size = image.size

    ### データを並べる
    data_1d = data.flatten()

    ### 対角行列を作る
    H_diag = np.eye(data_1d.shape[0])

    model = Lasso(alpha=0.0005).fit(H_diag, data_1d)
    image_raw_conv = model.coef_
    image_raw_conv = image_raw_conv.reshape(size[1], size[0])

    image_raw_conv_8bit = image_raw_conv - np.min(image_raw_conv) + 50 #255 * (image_raw_conv - np.min(image_raw_conv)) / (np.max(image_raw_conv) - np.min(image_raw_conv) )
    image_conv_8bit = Image.fromarray(np.uint8(image_raw_conv_8bit))
    image_conv_8bit.save(basepath + '\\export\\' + os.path.basename(filedir) + '_.bmp')


    plt.figure(figsize=(17,9))
    plt.subplot(221)
    plt.title('Input Image')
    plt.imshow(image, cmap='gray')


    plt.subplot(222)
    plt.title('Filtered Image by L1 norm')
    plt.imshow(image_raw_conv, cmap='gray')

    plt.subplot(223)
    plt.title('Histogram of Background')
    plt.bar(hist_bg[1][0:(len(hist_bg[0]))], hist_bg[0], width=1.0)

    #plt.subplot(234)
    #plt.imshow(data_reg, cmap='gray')

    plt.show()





pass

