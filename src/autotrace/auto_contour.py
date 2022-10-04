
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

basepath = '..\\..\\image\\autotrace\\'

with open(basepath + 'auto_contour.json', 'r') as f:
    tp_info = json.load(f)

imagedir = basepath + tp_info['reference']['viewed_image']
view_image = cv2.imread(imagedir)[:,:,0]


for filename, context in tp_info['sample'].items():
    data = np.loadtxt(basepath + filename + ".csv", delimiter = ',')

    index = 0
    for featurep in data:
        trim_c1 = (int(featurep[1]-featurep[3]/2), int(featurep[1]+featurep[3]/2))
        trim_c2 = (int(featurep[0]-featurep[2]/2), int(featurep[0]+featurep[2]/2))
        trim = view_image[trim_c1[0]:trim_c1[1], trim_c2[0]:trim_c2[1]].flatten()

        ### 対角行列を作る
        H_csr = lil_matrix((trim.shape[0], trim.shape[0]))
        for i in range(0, trim.shape[0]):
            H_csr[i, i] = 1

        alpha_val = featurep[4]
        trim = cv2.GaussianBlur(trim, (5,5), 2.0)
        # L1スパースモデリング
        model = Lasso(alpha=alpha_val).fit(H_csr, trim)
        trim = model.coef_.reshape(int(featurep[3]), int(featurep[2]))

        # 閾値処理
        hist_lasso = np.histogram( trim, range=(np.min(trim),np.max(trim)), bins=256)
        hist_lasso_max = hist_lasso[1][np.argmax(hist_lasso[0])]
        trim2 = np.where(trim < hist_lasso_max, trim, hist_lasso_max)

        trim2 = trim2 - np.min(trim2)
        trim2 = np.array(trim2, dtype=np.uint8)

        trim3 = cv2.Laplacian( trim2, cv2.CV_8U, ksize=5)

        # 形状検出
        contours, hierarchy = cv2.findContours( trim3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        
        save_trim = Image.fromarray(np.uint8(trim2))
        save_trim.save(basepath + '\\export\\' + filename + "_" + str(index) + '.bmp')


        fig = plt.figure()
        ax = plt.axes()

        plt.imshow(trim2, cmap='gray')

        # 作業データの出力　観測点群データへの変換
        with open(basepath + '\\export\\' + filename + '_convex_' + str(index) + '.csv', 'w') as f_out:
            cindex = 0
            for contour in contours:
                contour = contour.squeeze(axis=1)
                if cv2.contourArea(contour) / (trim3.shape[0] * trim3.shape[1])  < 0.01:
                    continue

                cenvec = np.array((trim3.shape[1]/2, trim3.shape[0]/2))
                if np.min(np.array([ np.linalg.norm(t,ord=1) for t in (contour - cenvec) ])) > np.linalg.norm(cenvec, ord=1) * 0.30:
                    continue

                cenvec = np.array( (trim3.shape[1] / 2, trim3.shape[0] / 2) )

                plt.scatter(contour[:,0], contour[:,1])
                ax.add_patch(patches.Rectangle(xy=(contour[int(len(contour)/4),0], contour[int(len(contour)/4),1]),width=10,height=10))
                ax.text(contour[int(len(contour)/4),0], contour[int(len(contour)/4),1], str(cindex), horizontalalignment='left', verticalalignment='top', size=12)

                for c in contour:
                    f_out.write( format(c[0], '.4f') + ',' + format(c[1], '.4f') + ',' + str(cindex) + '\n')

                cindex = cindex + 1


        fig.savefig(basepath + '\\export\\' + filename + "_fig_" + str(index) + '.png')
        #plt.show()

        print(str(index) + " Finished")
        index = index + 1
        plt.close()





