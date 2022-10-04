
import numpy as np

from PIL import Image
from sklearn.linear_model import Lasso, Ridge
from scipy.sparse import csr_matrix, lil_matrix

import cv2
import itertools
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import glob

import json

basepath = '..\\..\image\\practice\\'

with open(basepath + 'tracing.json', 'r') as f:
    tp_info = json.load(f)



### サンプル画像
for key, value in tp_info['sample'].items():
    # サンプル画像読込
    image_sample = cv2.imread(basepath + '\\tp\\' + key + '.bmp')
    filtered_sample = cv2.Laplacian( image_sample, cv2.CV_8U, ksize=5)

    trim_sample = filtered_sample[:, :, 0]
    contours, hierarchy = cv2.findContours( trim_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(str(cv2.contourArea(contour)))

    plt.figure(figsize=(17,9))

    ax = plt.subplot(131)
    plt.title('Input Image')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(image_sample, cmap='gray')

    ax = plt.subplot(132)
    plt.title('Input Image(Laplacian)')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(filtered_sample, cmap='gray')
    
    cv2.imwrite(basepath + '\\export\\' + key + '_.bmp', filtered_sample)
    
    
    ax = plt.subplot(133)
    plt.title('With Contour')
    plt.imshow(trim_sample, cmap='gray')
    
    color_flag = 0
    dist_neighbor = 0.0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10:
            continue

        convexes = cv2.convexHull(cnt)
        #cnt_approx = cv2.approxPolyDP(cnt, 1, True)

        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)

        # 各輪郭点をラベリングする
        cnt_grouping = [-1] * len(cnt)
        cnt_grouped = {}
        label = 0
        for i in range(0, len(cnt_grouping)):
            if cnt_grouping[i] != -1:
                continue
            cnt_grouping[i]  = label
            cnt_grouped[label] = [cnt[i]]

            for j in range(i+1, len(cnt_grouping)):
                if cnt_grouping[j] == -1:
                    diff_v = np.array(([cnt[i, 0], cnt[i,1]])) - np.array(([cnt[j, 0], cnt[j,1]]))

                    dist = np.linalg.norm(diff_v)
                    if dist < dist_neighbor:
                        cnt_grouping[j] = label
                        cnt_grouped[label].append(cnt[j])


            label += 1

        points_view_grouped = []
        for k, v in cnt_grouped.items():
            point = (np.average([t[0] for t in v]), np.average([t[1] for t in v]))
            points_view_grouped.append(point)

        """
        # 抽出したベクトルの中点と法線ベクトルを演算する
        cnt_mid = np.zeros( (len(cnt)-1, 2) )
        norm_vect = np.zeros( (len(cnt)-1, 2) )
        for i in range(0, len(cnt)-1):
            vec1 = np.array((cnt[i][0], cnt[i][1], 0))
            vec2 = np.array((cnt[i+1][0], cnt[i+1][1], 0))
            n = np.cross(vec2 - vec1, np.array((0,0,1)))
            n = n[0:2] / np.linalg.norm(n)
            cnt_mid[i] = (vec1[0:2] + vec2[0:2]) / 2
            norm_vect[i] = n
        """

        #cnt_approx = cnt_approx.squeeze(axis=1)

        # 輪郭の点同士を結ぶ線を描画する。
        """
        for convex in convexes:
            if color_flag == 0:
                ax.add_patch(patches.Ellipse(xy=(convex[0,0], convex[0,1]), width=2, height=2, fc='b'))
            else:
                ax.add_patch(patches.Ellipse(xy=(convex[0,0], convex[0,1]), width=2, height=2, fc='r'))    
        """
        with open(basepath + '\\export\\' + key + '_convex' + str(color_flag) + '.txt', 'w') as f:
            for k, v in cnt_grouped.items():
                f.write( format(np.average([t[0] for t in v]), '.4f') + ',' + format(np.average([t[1] for t in v]), '.4f') + ',' + str(k) + '\n')
                
            #for i in range(0, len(cnt)):
            #    f.write(str(cnt[i][0]) + ',' + str(cnt[i][1]) + ',' + str(cnt_grouping[i]) + '\n')

        color = "r"
        if color_flag == 0:        
            color = "b"
            
            #ax.add_patch(plt.Polygon(points_view_grouped, color="b", fill=None, lw=2))
            #ax.add_patch(plt.Polygon(cnt_approx, color="b", fill=None, lw=2))
        else:
            pass
            #ax.add_patch(plt.Polygon(points_view_grouped, color="r", fill=None, lw=2))
            #ax.add_patch(plt.Polygon(cnt_approx, color="r", fill=None, lw=2))


        plt.scatter(cnt[:,0], cnt[:,1])

        ax.add_patch(patches.Rectangle(xy=(cnt[int(len(cnt)/4),0], cnt[int(len(cnt)/4),1]),width=10,height=10))
        ax.text(cnt[int(len(cnt)/4),0], cnt[int(len(cnt)/4),1], str(color_flag), horizontalalignment='left', verticalalignment='top', size=12)
        #for p in points_view_grouped:
        #    ax.add_patch(patches.Ellipse(xy=(p[0], p[1]), width=2, height=2, fc=color))

        color_flag = color_flag + 1
        

    plt.show()
