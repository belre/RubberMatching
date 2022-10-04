
import numpy as np
import cv2
import json

import matplotlib.pyplot as plt

basepath = '..\\..\image\\practice\\'

with open(basepath + 'polar_image.json', 'r') as f:
    tp_info = json.load(f)


datalen = 1800
deglen = 360*16

for filename, value in tp_info.items():
    img = cv2.imread(basepath + '\\' + filename + '.bmp')

    center0 = np.array((img.shape[0]/2, img.shape[1]/2))

    conv_img = np.zeros((deglen, datalen))
    for j in range(0, deglen):
        for i in range(0, datalen):
            degi = np.pi * 360 * j / (deglen * 180)

            pnt = np.array((i * np.cos(degi), (i * np.sin(degi)))) + center0
            if pnt[0] >= 0 and pnt[0] < img.shape[0] and pnt[1] >= 0 and pnt[1] < img.shape[1]:
                conv_img[j, i] = img[ int(pnt[1]), int(pnt[0]), 0]

    plt.figure()
    plt.imshow(conv_img, cmap='gray')
    plt.show()

    





