
import json
import matplotlib.pyplot as plt

import cv2
import numpy as np

basepath = '..\\..\image\\practice\\'

with open(basepath + 'noiseremoval.json', 'r') as f:
    tp_info = json.load(f)

for ck, cv in tp_info.items():
    image_sample = cv2.imread(basepath + ck + '.jpg')
    #image_sample = np.log(image_sample+1)
    image_sample = cv2.GaussianBlur(image_sample, (5,5), 1.0)
    image_sample = cv2.fastNlMeansDenoising( image_sample, None, h=40)
    image_sample = cv2.Sobel( image_sample, cv2.CV_8U, 0, 1, ksize=5)
    #image_sample = cv2.Laplacian(image_sample, cv2.CV_8U, ksize=5)

    plt.figure(figsize=(17,9))
    plt.imshow(image_sample)
    
    plt.show()