
import numpy as np

from PIL import Image
from sklearn.linear_model import Lasso, Ridge
from scipy.sparse import csr_matrix, lil_matrix
from scipy import signal

import cv2
import itertools
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import glob

import json

basepath = '..\\..\\image\\matching\\'

with open(basepath + 'compare_contour.json', 'r') as f:
    tp_info = json.load(f)


with open(basepath + 'shapetemplate.json', 'r') as f:
    shape_tp_info = json.load(f)


### テンプレート画像を読込
image_tp = cv2.imread(basepath + tp_info['reference']['filepath'])
image_view_tp = cv2.imread(basepath + tp_info['reference']['imgpath'])
roi_list = tp_info['reference']['roi']

# roi保存
for name, roi in roi_list.items():
    image_tp_roi = image_view_tp[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.imwrite(basepath + tp_info['reference']['imgpath'] + "_" + name + ".bmp", image_tp_roi)
    pass

def poly_model(solv, x, degree):
    xl = np.array([x ** d for d in range(degree, -1, -1)])
    return solv @ xl

def draw_shape(posit, coord_dict):
    for ck, cv in shape_tp_info.items():
        roi = cv["region"]
        if posit != None:
            ax.add_patch(patches.Rectangle(xy=(roi[0]+posit[ck][0], roi[1]+posit[ck][1]), width=roi[2]-roi[0], height=roi[3]-roi[1], ec='g', fill=False))
        else:
            ax.add_patch(patches.Rectangle(xy=(roi[0], roi[1]), width=roi[2]-roi[0], height=roi[3]-roi[1], ec='g', fill=False))

        param = cv["param"]
        if cv["type"] == "circle":            
            if posit != None:
                ax.add_patch(patches.Arc(xy=(param[0][0]+posit[ck][0],param[0][1]+posit[ck][1]), width=param[1]*2, height=param[1]*2, fc='None', ec='red', theta1=np.rad2deg(param[2]), theta2=np.rad2deg(param[3])))
            else:
                ax.add_patch(patches.Arc(xy=param[0], width=param[1]*2, height=param[1]*2, fc='None', ec='red', theta1=np.rad2deg(param[2]), theta2=np.rad2deg(param[3])))    
        elif cv['type'] == 'verticalline':
            y = np.linspace(param[1], param[2])
            if posit != None:
                plt.plot( param[0][0] * y + param[0][1] + posit[ck][0], y + posit[ck][1], color='r')
            else:
                plt.plot( param[0][0] * y + param[0][1], y, color='r')
        elif cv['type'] == 'hline':
            x = np.linspace(param[1], param[2])
            if posit != None:
                plt.plot( x + posit[ck][0], param[0][0] * x + param[0][1]+posit[ck][1], color='r')
            else:
                plt.plot( x, param[0][0] * x + param[0][1], color='r')
        elif cv['type'] == 'hpoly':
            x = np.linspace(param[1], param[2])
            
            if posit != None:
                plt.plot( x + posit[ck][0], poly_model(param[0], x, 2) + posit[ck][1],  color='r')
            else:
                plt.plot( x, poly_model(param[0], x, 2), color='r')
        elif cv['type'] == 'vpoly':
            y = np.linspace(param[1], param[2])

            if posit != None:
                plt.plot( poly_model(param[0], y, 2) + posit[ck][0], y + posit[ck][1], color='r')
            else:
                plt.plot( poly_model(param[0], y, 2), y, color='r')

        if coord_dict != None:
            coord_list = coord_dict[ck]
            plt.scatter( coord_list[:, 0], coord_list[:, 1], color='g')
            plt.plot( coord_list[:, 2], coord_list[:, 3], color='g', linestyle="dashed")
            plt.plot( coord_list[:, 4], coord_list[:, 5], color='g', linestyle="dashed")




# 直線の法線ベクトルに沿って
# 画素データを抽出する
def getpix(pnt, param_, drange):
    pix_list = np.zeros((len(pnt), drange[1] - drange[0] + 1))
    for i in range( 0, len(pnt) ):
        x = pnt[i]
        y = param_[0][0] * pnt[i] + param_[0][1]
        normal_v = np.array((1, - 1 / param_[0][0]))
        normal_v /= np.linalg.norm(normal_v)

        nr = np.linspace(drange[0], drange[1], num=drange[1] - drange[0] + 1)
        pix = np.empty((drange[1] - drange[0] + 1))
        for j in range(0, nr.shape[0]):
            nv = normal_v * nr[j]
            nv[0] = nv[0] + x + part_result[ck][0] - region[0] + posit_error
            nv[1] = nv[1] + y + part_result[ck][1] - region[1] + posit_error
            pix[j] = image_sample_p_n[int(nv[1]), int(nv[0]), 0]
                    
        pix_list[i,:] = pix
    return pix_list

# 直線に合わせる
def adjust_line(param):
    # 相互相関係数を使って1D位置マッチング
    drange = (-60, 60)
    drrange = (-50, 50)
    pnt_c = np.linspace(param[1], param[2])

    pix_list = getpix(pnt_c, param, drange)

    pix_ref = pix_list[ int(pix_list.shape[0]/2), int((drange[1] - drange[0])/2+drrange[0]):int((drange[1] - drange[0])/2+drrange[1])]

    # 位置修正後のx,yを算出
    match_pnt = np.empty((len(pnt_c), 3))
    for i in range( 0, len(pnt_c) ):
        x = pnt_c[i]
        y = param[0][0] * pnt_c[i] + param[0][1]
        corr = np.correlate( pix_list[i,:] - np.average(pix_list[i,:]), pix_ref - np.average(pix_ref), "full")
        corrmax = np.argmax(corr) - (drange[1] + drrange[1] - drange[0] - drrange[0]) / 2
        match_pnt[i] = (x, y, y + corrmax)
        print(str(int(x)) + "," + str(int(y)) + "," + str(corrmax))

    # 新たな切片と傾きを演算
    solv = np.polyfit(match_pnt[:,0], match_pnt[:,2], 1)
    param[0][0] = solv[0]
    param[0][1] = solv[1]

    # 傾きを最適値に併せる
    pix_list = getpix(pnt_c, param, drange)
    pix_stat = np.empty((drange[1] - drange[0] + 1, 2))

    for i in range( 0, drange[1] - drange[0] + 1):
        pix_stat[i,0] = np.average(pix_list[:, i])
        pix_stat[i,1] = np.var(pix_list[:, i])

    # ピーク検出
    pix_args_min = (np.array( signal.argrelmin(pix_stat[:, 0], order=5)) - (drange[1] - drange[0] ) / 2)[0]

    # 0に近いargsの2つを抽出して、切片を補正
    args_min1 = np.argmin(np.abs(pix_args_min))
    pix_args_min1 = pix_args_min[args_min1]
    pix_args_min2 = pix_args_min[np.argmin(np.abs( np.delete(pix_args_min, args_min1)))]
    param[0][1] += (pix_args_min1 + pix_args_min2) / 2

    return param


### サンプル画像
for item in tp_info['sample']:

    # サンプル画像読込
    image_sample = cv2.imread(basepath + item + '.bmp')
    image_sample = cv2.GaussianBlur( image_sample, (1,1), 0)

    # パターンマッチング(全体)
    result = cv2.matchTemplate( image_sample, image_tp, cv2.TM_CCOEFF)
    loc_result = cv2.minMaxLoc(result)

    # パターンマッチング(部分)
    part_result = {}
    posit_error = 70
    for ck, cv in shape_tp_info.items():
        if 'region' in cv:
            # 領域の定義
            region = cv['region']
            image_ref_p = cv2.imread(basepath + '.\\tp\\' + cv['tppath'])
                
            # マスク画像
            image_mask_p = None
            if 'maskpath' in cv:
                image_mask_p = cv2.imread(basepath + '.\\tp\\' + cv['maskpath'])

            image_sample_p = image_sample[(loc_result[3][1]+region[1]-posit_error):(loc_result[3][1]+region[3]+posit_error), (loc_result[3][0]+region[0]-posit_error):(loc_result[3][0]+region[2]+posit_error)]
            image_sample_p_n = cv2.fastNlMeansDenoising(image_sample_p, None, h=10)
            
            #image_sample_p_n = cv2.GaussianBlur(image_sample_p_n[:,:], (11,11), 1.0)           

            if 'ignore_matching' in cv and cv['ignore_matching'] == True:
                part_result[ck] = [0, 0]
            else:
                # マッチングの結果
                result_p = cv2.matchTemplate( image_sample_p_n, image_ref_p, cv2.TM_CCOEFF, mask=image_mask_p)
                result_p = cv2.minMaxLoc(result_p)
                part_result[ck] = [result_p[3][0] - posit_error, result_p[3][1] - posit_error ]

            # 直線の位置修正
            if cv['type'] == 'hline':
                param = cv["param"]

                param = adjust_line(param)
                shape_tp_info[ck]['param'] = param

    # 直線に従って、リップ部の射影データをくりぬく
    # リップ部の特徴点と法線ベクトル間の2つの点を算出する
    rip_width = 36
    coord_dict = {}
    for ck, cv in shape_tp_info.items():
        param = cv['param']
        posit = part_result[ck]

        div_points = 50
        coord_list = np.zeros((div_points, 6))          # 前半は座標, 後半は法線ベクトル

        if cv["type"] == "circle":      
            centerp = param[0][0]+posit[0],param[0][1]+posit[1]

            div_points = int(param[1] * (2 * np.pi + param[3] - param[2]))
            coord_list = np.zeros((div_points, 6)) 

            for i in range(0, div_points):
                theta = i * (2*np.pi + param[3] - param[2]) / div_points + param[2]
                coord_list[i, 0] = centerp[0] + param[1] * np.cos(theta)
                coord_list[i, 1] = centerp[1] + param[1] * np.sin(theta)

                normal_vec = np.cos(theta), np.sin(theta)
                coord_list[i, 2] = coord_list[i, 0] + normal_vec[0] * (-rip_width / 2)
                coord_list[i, 3] = coord_list[i, 1] + normal_vec[1] * (-rip_width / 2)
                coord_list[i, 4] = coord_list[i, 0] + normal_vec[0] * (+rip_width / 2)
                coord_list[i, 5] = coord_list[i, 1] + normal_vec[1] * (+rip_width / 2)

        elif cv['type'] == 'verticalline':
            div_points = int((param[2] - param[1]) * np.sqrt(1 + param[0][0] ** 2))
            coord_list = np.zeros((div_points, 6)) 

            for i in range(0, div_points):
                pm_y = i * (param[2] - param[1]) / div_points + param[1]+ posit[1]
                x = param[0][0] * pm_y + param[0][1] + posit[0]
                coord_list[i, 0] = x
                coord_list[i, 1] = pm_y

                normal_vec = -1 / param[0][0], 1
                normal_vec /= np.linalg.norm(normal_vec)
                coord_list[i, 2] = coord_list[i, 0] + normal_vec[0] * (-rip_width / 2)
                coord_list[i, 3] = coord_list[i, 1] + normal_vec[1] * (-rip_width / 2)
                coord_list[i, 4] = coord_list[i, 0] + normal_vec[0] * (+rip_width / 2)
                coord_list[i, 5] = coord_list[i, 1] + normal_vec[1] * (+rip_width / 2)
        elif cv['type'] == 'hline':
            div_points = int((param[2] - param[1]) * np.sqrt(1 + param[0][0] ** 2))
            coord_list = np.zeros((div_points, 6)) 

            for i in range(0, div_points):
                pm_x = i * (param[2] - param[1]) / div_points + param[1] + posit[0]
                y = param[0][0] * pm_x + param[0][1] + posit[1]
                coord_list[i, 0] = pm_x
                coord_list[i, 1] = y
            
                normal_vec = 1, -1 / param[0][0]
                normal_vec /= np.linalg.norm(normal_vec)
                coord_list[i, 2] = coord_list[i, 0] + normal_vec[0] * (-rip_width / 2)
                coord_list[i, 3] = coord_list[i, 1] + normal_vec[1] * (-rip_width / 2)
                coord_list[i, 4] = coord_list[i, 0] + normal_vec[0] * (+rip_width / 2)
                coord_list[i, 5] = coord_list[i, 1] + normal_vec[1] * (+rip_width / 2)

        elif cv['type'] == 'hpoly':
            theta_int = np.arctan(2 * param[0][0] * param[1] + param[0][1]), np.arctan(2 * param[0][0] * param[2] + param[0][1])
            f_int = np.sin(theta_int) / (2 * np.cos(theta_int) ** 2) + np.log((1+np.cos(theta_int) / (1-np.sin(theta_int)))) / 4
            f_int /= 2 * param[0][0]
            div_points = int( np.abs(f_int[1] - f_int[0]))
            coord_list = np.zeros((div_points, 6)) 

            last_normal_vec = np.array((0, 0))
            for i in range(0, div_points):     
                pm_x = i * (param[2] - param[1]) / div_points + param[1] 
                y = poly_model(param[0], pm_x, 2) 
                coord_list[i, 0] = pm_x + posit[0]
                coord_list[i, 1] = y + posit[1]

                # 法線ベクトルの連続性を前データと比べて維持する
                normal_vec = 1, - 1 / (2 * param[0][0] * pm_x + param[0][1])
                normal_vec /= np.linalg.norm(normal_vec)
                normal_vec = normal_vec if normal_vec @ last_normal_vec >= 0 else -normal_vec
                last_normal_vec = normal_vec

                coord_list[i, 2] = coord_list[i, 0] + normal_vec[0] * (-rip_width / 2)
                coord_list[i, 3] = coord_list[i, 1] + normal_vec[1] * (-rip_width / 2)
                coord_list[i, 4] = coord_list[i, 0] + normal_vec[0] * (+rip_width / 2)
                coord_list[i, 5] = coord_list[i, 1] + normal_vec[1] * (+rip_width / 2)

        elif cv['type'] == 'vpoly':
            theta_int = np.arctan(2 * param[0][0] * param[1] + param[0][1]), np.arctan(2 * param[0][0] * param[2] + param[0][1])
            f_int = np.sin(theta_int) / (2 * np.cos(theta_int) ** 2) + np.log((1+np.cos(theta_int) / (1-np.sin(theta_int)))) / 4
            f_int /= 2 * param[0][0]
            div_points = int( np.abs(f_int[1] - f_int[0]))
            coord_list = np.zeros((div_points, 6)) 

            last_normal_vec = np.array((0, 0))
            for i in range(0, div_points):
                pm_y = i * (param[2] - param[1]) / div_points + param[1]
                x = poly_model(param[0], pm_y, 2)
                coord_list[i, 0] = x
                coord_list[i, 1] = pm_y

                # 法線ベクトルの連続性を前データと比べて維持する
                normal_vec = - 1 / (2 * param[0][0] * pm_y + param[0][1]), 1
                normal_vec /= np.linalg.norm(normal_vec)
                normal_vec = normal_vec if normal_vec @ last_normal_vec >= 0 else -normal_vec
                last_normal_vec = normal_vec

                coord_list[i, 2] = coord_list[i, 0] + normal_vec[0] * (-rip_width / 2)
                coord_list[i, 3] = coord_list[i, 1] + normal_vec[1] * (-rip_width / 2)
                coord_list[i, 4] = coord_list[i, 0] + normal_vec[0] * (+rip_width / 2)
                coord_list[i, 5] = coord_list[i, 1] + normal_vec[1] * (+rip_width / 2)
        
        coord_dict[ck] = coord_list


    image_sample_trim = cv2.fastNlMeansDenoising(image_sample[loc_result[3][1]:loc_result[3][1]+image_tp.shape[0],loc_result[3][0]:loc_result[3][0]+image_tp.shape[1]], None, h=10)



    # 射影データに従って、データを射影する
    # 射影したデータを保存する
    for ck, cv in shape_tp_info.items():
        coord_list = coord_dict[ck]

        pix_list = np.zeros((len(coord_list), rip_width))
        for i in range(0, coord_list.shape[0]):
            tan_vec = np.array( (coord_list[i, 4] - coord_list[i, 2], coord_list[i, 5] - coord_list[i, 3]))
            tan_vec /= np.linalg.norm(tan_vec)

            lin = np.empty((rip_width, 2))
            for j in range(0, rip_width):
                posv = np.array((coord_list[i, 2], coord_list[i, 3])) + tan_vec * j

                pix_list[i, j] = image_sample_trim[ int(posv[1]), int(posv[0]), 0]
                lin[j] = posv

        pix_list = cv2.Sobel( pix_list, cv2.CV_8U, 1, 0, ksize=5)
        
        im = Image.fromarray(pix_list).convert('RGB')
        im.save(basepath + "\\export\\" + item + "_" + ck + '_linearize.bmp')
        

        #plt.figure(figsize=(17, 9))
        #plt.title('trace:' + ck)
        #plt.imshow(pix_list, cmap='gray')
        #plt.imshow(image_sample_trim)
        #plt.scatter(lin[:, 0], lin[:, 1])
        #plt.show()
        




    #image_sample_trim = cv2.fastNlMeansDenoising(image_sample[loc_result[3][1]:loc_result[3][1]+image_tp.shape[0],loc_result[3][0]:loc_result[3][0]+image_tp.shape[1]], None, h=15)


    plt.figure(figsize=(17,9))

    ax = plt.subplot(131)
    plt.title('Template Image')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(image_tp, cmap='gray')
    draw_shape(None, None)


    ax = plt.subplot(132)
    plt.title('Template Image(VIEW)')
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    plt.imshow(image_view_tp, cmap='gray')
    draw_shape(None, None)

    
    ax = plt.subplot(133)
    plt.title('Input Image (After pre-positioning)')
    plt.imshow(image_sample_trim , cmap='gray')  
    plt.xlabel('X (x 1/8)')
    plt.ylabel('Y (x 1/8)')
    draw_shape(part_result, coord_dict)

    plt.show()
