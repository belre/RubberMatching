import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import comb
from scipy.sparse import lil_matrix
from sklearn.linear_model import Lasso
import numpy as np
import cv2
import os

basepath = '..\\..\\image\\autotrace\\'

dist_th = 8*10


with open(basepath + 'auto_contour.json', 'r') as f:
    tp_info = json.load(f)

imagedir = basepath + tp_info['reference']['viewed_image']
view_image = cv2.imread(imagedir)

pcddir = basepath + tp_info['pcd']['dir']


def fit_round(plist, ref_ptr):
    # 円中心の行列を作成.
    plist = np.array(plist) 
    plist_sq = plist * plist
    plist_tri = plist * plist * plist
    plist_cov = plist[:,0] * plist[:,1]
    plist_covxy2 = plist[:, 0] * plist[:,1] * plist[:,1]
    plist_covx2y = plist[:, 0] * plist[:,0] * plist[:,1]

    # 円中心を構成する行列式を設計
    mat_round = np.array( (( np.sum(plist_sq[:,0]), np.sum(plist_cov[:]), np.sum(plist[:,0])),
                                (np.sum(plist_cov[:]), np.sum(plist_sq[:,1]), np.sum(plist[:,1])),
                                (np.sum(plist[:,0]), np.sum(plist[:,1]), len(plist))))
    bvec = np.array( ( -np.sum(plist_tri[:,0]) - np.sum(plist_covxy2), -np.sum(plist_tri[:,1]) - np.sum(plist_covx2y), -np.sum(plist_sq[:,0]) - np.sum(plist_sq[:,1]) ))
    solv = np.dot(np.linalg.inv(mat_round), bvec)
    center_round = np.array( (-solv[0] / 2, -solv[1] / 2))
    radius = np.sqrt(np.dot(center_round, center_round) - solv[2])

    # plistを円中心に変換
    plist_conv = plist[:] - center_round

    # 円弧の角度を算出する
    dec_to_y1 = 0
    dec_to_y2 = 0        
    for local_deg in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        cos0 = np.cos(local_deg)
        sin0 = np.sin(local_deg) 

        deg_list1 = np.array([np.arctan2( cos0 * xy[1] - sin0 * xy[0], cos0 * xy[0] + sin0 * xy[1]) for xy in plist_conv[:]  if cos0 * xy[1] - sin0 * xy[0] > 0  ])
        deg_list2 = np.array([np.arctan2( cos0 * xy[1] - sin0 * xy[0], cos0 * xy[0] + sin0 * xy[1]) for xy in plist_conv[:]  if cos0 * xy[1] - sin0 * xy[0] < 0  ])

        if np.min(deg_list1) - np.max(deg_list2)  < np.pi / 2:
            pass
        else:
            dec_to_y1 = np.min(deg_list1) + local_deg
            dec_to_y2 = np.max(deg_list2) + local_deg
            break

    center_round[0] -= ref_ptr[0]
    center_round[1] -= ref_ptr[1]

    return center_round, radius, dec_to_y1, dec_to_y2

def fit_line(plist, ref_ptr):
    # 行列の作成
    plist_sq = plist * plist
    plist_cov_xy = plist[:, 0] * plist[:, 1]
    
    mat_line = np.array( (( np.sum(plist_sq[:,0]), np.sum(plist[:,0]) ), ( np.sum(plist[:,0]), len(plist))))
    bvec = np.array((np.sum(plist_cov_xy), np.sum(plist[:, 1])))
    solv = np.linalg.inv(mat_line) @ bvec

    # エラー量を比較して
    # 変域を設定
    typical_pnt = {}
    for data in plist:
        if data[0] not in typical_pnt:
            typical_pnt[data[0]] = data[1]
        else:
            newerr = data[1] - (solv[0] * data[0] + solv[1])
            olderr = typical_pnt[data[0]] - (solv[0] * data[0] + solv[1])
            if np.abs(newerr) < np.abs(olderr):
                typical_pnt[data[0]] = data[1]

    solv[1] = solv[1] - ref_ptr[1] + solv[0] * ref_ptr[0]

    return solv, np.min(list(typical_pnt.keys())) - ref_ptr[0], np.max(list(typical_pnt.keys())) - ref_ptr[0]

import sympy

def fit_poly(plist, degree, ref_ptr):
    def model(solv, x):
        xl = np.array([x ** d for d in range(degree, -1, -1)])
        return solv @ xl #solv[0] * x ** 2 + solv[1] * x + solv[2]

    def tr_mat():
        mat = np.identity(degree + 1)
        for i in range(0, degree+1, 1):         # degree～0
            for j in range(0, i):                   #degree～k+1
                degi = degree - i
                degj = degree - j
                mat[i, j] = comb(degj, degi) * ref_ptr[0] **(degj - degi)
                pass

        return mat

    solv = np.polyfit(plist[:,0], plist[:,1], degree)


    # エラー量を比較して
    # 変域を設定
    typical_pnt = {}
    for data in plist:
        if data[0] not in typical_pnt:
            typical_pnt[data[0]] = data[1]
        else:
            newerr = data[1] - model(solv, data[0])
            olderr = typical_pnt[data[0]] - model(solv, data[0])
            if np.abs(newerr) < np.abs(olderr):
                typical_pnt[data[0]] = data[1]

    solvr = tr_mat() @ solv
    solvr[len(solvr)-1] -= ref_ptr[1]

    return solvr, np.min(list(typical_pnt.keys())) - ref_ptr[0], np.max(list(typical_pnt.keys())) - ref_ptr[0], model



for filename, context in tp_info['sample'].items():
    area = np.loadtxt(basepath + filename + ".csv", delimiter = ',')
    length = len(area)

    data = {}
    for i in range(0, length):
        data_t = np.loadtxt(pcddir + '\\' + os.path.basename(filename).split('.', 1)[0] + '_convex_' + str(i) + '.csv', delimiter = ',')
        data[i] = data_t

    # ROIごとにデータをまとめる
    data_save_roi = {}
    fit_items = {}
    for ck, cv in context.items():
        fit_items[ck] = {}
        data_save_roi[ck] = {}
        #data_roi[ck] = np.empty((0,2))
    
    # データの登録
    for ck, cv in context.items():
        datalist = np.empty((0,2))
        for i, v in data.items():
            if i in cv['number']:
                vv = np.transpose((v[:,0] + area[i,0] - area[i,2] / 2, v[:,1] + area[i,1] - area[i,3] / 2))
                datalist = np.vstack( ( datalist, np.array(vv)))
            
        # 重複行の排除
        # https://qiita.com/uuuno/items/b714d84ca2edbf16ea19
        #datalist = np.array(list(map(list, set(map(tuple, datalist)))))



        if cv['type'] == 'circle':
            center, radius, theta1, theta2 = fit_round(datalist, tp_info['reference']['base_coord'])
            fit_items[ck]['param'] = np.array( (center, radius, theta1, theta2) )
            fit_items[ck]['type'] = 'circle'

            data_save_roi[ck]['param'] = [center.tolist(), radius, theta1, theta2]

            xs = int(center[0]-radius*1.5)
            xe = int(center[0]+radius*1.5)
            ys = int(center[1]-radius*1.5)
            ye = int(center[1]+radius*1.5)
            data_save_roi[ck]['region'] = [xs, ys, xe, ye]
        elif cv['type'] == 'verticalline':
            tmp = np.transpose(np.vstack( (datalist[:, 1], datalist[:, 0])))
            base_tr = [ tp_info['reference']['base_coord'][1], tp_info['reference']['base_coord'][0], tp_info['reference']['base_coord'][3], tp_info['reference']['base_coord'][2]]

            fit_items[ck]['param'] = fit_line(tmp, base_tr)
            fit_items[ck]['type'] = 'verticalline'

            data_save_roi[ck]['param'] = [fit_items[ck]['param'][0].tolist(), fit_items[ck]['param'][1], fit_items[ck]['param'][2]]
            data_save_roi[ck]['param'] = [fit_items[ck]['param'][0].tolist(), fit_items[ck]['param'][1], fit_items[ck]['param'][2]]
            x1 = int(fit_items[ck]['param'][0][0] * fit_items[ck]['param'][1] + fit_items[ck]['param'][0][1]) 
            x2 = int(fit_items[ck]['param'][0][0] * fit_items[ck]['param'][2] + fit_items[ck]['param'][0][1])
            y1 = int(fit_items[ck]['param'][1])
            y2 = int(fit_items[ck]['param'][2])
            data_save_roi[ck]['region'] = [ (x1 if x1 < x2 else x2) - 50, y1 if y1 < y2 else y2, (x2 if x1 < x2 else x1) + 50, y2 if y1 < y2 else y1]
        elif cv['type'] == 'hline':
            fit_items[ck]['param'] = fit_line(datalist, tp_info['reference']['base_coord'])
            fit_items[ck]['type'] = 'hline'  

            data_save_roi[ck]['param'] = [fit_items[ck]['param'][0].tolist(), fit_items[ck]['param'][1], fit_items[ck]['param'][2]]
            x1 = int(fit_items[ck]['param'][1])
            x2 = int(fit_items[ck]['param'][2])
            y1 = int(fit_items[ck]['param'][0][0] * fit_items[ck]['param'][1] + fit_items[ck]['param'][0][1])
            y2 = int(fit_items[ck]['param'][0][0] * fit_items[ck]['param'][2] + fit_items[ck]['param'][0][1])
            data_save_roi[ck]['region'] = [ x1 if x1 < x2 else x2, (y1 if y1 < y2 else y2) - 50, x2 if x1 < x2 else x1, (y2 if y1 < y2 else y1) + 50]
        elif cv['type'] == 'hpoly':
            fit_items[ck]['param'] = fit_poly(datalist, cv['degree'], tp_info['reference']['base_coord'])
            fit_items[ck]['model'] = fit_items[ck]['param'][len(fit_items[ck]['param'])-1]
            fit_items[ck]['type'] = 'hpoly'
            data_save_roi[ck]['param'] = [fit_items[ck]['param'][0].tolist(), fit_items[ck]['param'][1], fit_items[ck]['param'][2]]   

            x1 = int(fit_items[ck]['param'][1])
            x2 = int(fit_items[ck]['param'][2])
            y1 = int(fit_items[ck]['model'](fit_items[ck]['param'][0], fit_items[ck]['param'][1]))
            y2 = int(fit_items[ck]['model'](fit_items[ck]['param'][0], fit_items[ck]['param'][2]))
            data_save_roi[ck]['region'] = [ x1 if x1 < x2 else x2, (y1 if y1 < y2 else y2), x2 if x1 < x2 else x1, (y2 if y1 < y2 else y1)]
        elif cv['type'] == 'vpoly':
            tmp = np.transpose(np.vstack( (datalist[:, 1], datalist[:, 0])))
            base_tr = [ tp_info['reference']['base_coord'][1], tp_info['reference']['base_coord'][0], tp_info['reference']['base_coord'][3], tp_info['reference']['base_coord'][2]]

            fit_items[ck]['param'] = fit_poly(tmp, cv['degree'], base_tr)
            fit_items[ck]['model'] = fit_items[ck]['param'][len(fit_items[ck]['param'])-1]
            fit_items[ck]['type'] = 'vpoly'
            data_save_roi[ck]['param'] = [fit_items[ck]['param'][0].tolist(), fit_items[ck]['param'][1], fit_items[ck]['param'][2]]            

            x1 = int(fit_items[ck]['model'](fit_items[ck]['param'][0], fit_items[ck]['param'][1]))
            x2 = int(fit_items[ck]['model'](fit_items[ck]['param'][0], fit_items[ck]['param'][2]))
            y1 = int(fit_items[ck]['param'][1])
            y2 = int(fit_items[ck]['param'][2])
            data_save_roi[ck]['region'] = [ x1 if x1 < x2 else x2, (y1 if y1 < y2 else y2), x2 if x1 < x2 else x1, (y2 if y1 < y2 else y1)]


        data_save_roi[ck]['tppath'] = "tp_" + filename + "_" + ck + ".bmp"
        data_save_roi[ck]['type'] = cv['type']
        #data_save_roi[ck]['param'] = fit_items[ck]['param'].tolist()

        

        fit_items[ck]['raw'] = datalist



    ####
    plt.figure(figsize=(17,7))
    plt.title(filename)

    # 現画像+フィッティング画像
    #ax = plt.subplot(131)
    ax = plt.subplot(111)
    basec = tp_info['reference']['base_coord']
    view_image_trim = view_image[basec[1]:basec[1]+basec[3], basec[0]:basec[0]+basec[2]]
    view_image_trim = cv2.cvtColor(view_image_trim, cv2.COLOR_BGR2GRAY) 
    plt.imshow(view_image_trim, cmap='gray')
    #plt.xlim(0, view_image.shape[1])
    #plt.ylim(0, view_image.shape[0])
    for roiname, item in fit_items.items():
        if 'type' not in item:
            continue

        if item['type'] == 'circle':
            ax.add_patch(patches.Arc(xy=(item['param'][0]), width=item['param'][1]*2, height=item['param'][1]*2, fc='None', ec='red', theta1=np.rad2deg(item['param'][2]), theta2=np.rad2deg(item['param'][3])))
        elif item['type'] == 'verticalline':
            y = np.linspace(item['param'][1], item['param'][2])
            plt.plot( item['param'][0][0] * y + item['param'][0][1], y, color='r')
        elif item['type'] == 'hline':
            x = np.linspace(item['param'][1], item['param'][2])
            plt.plot( x, item['param'][0][0] * x + item['param'][0][1], color='r')
        elif item['type'] == 'hpoly':
            x = np.linspace(item['param'][1], item['param'][2])
            plt.plot( x, item['model'](item['param'][0], x), color='r')
        elif item['type'] == 'vpoly':
            y = np.linspace(item['param'][1], item['param'][2])
            plt.plot( item['model'](item['param'][0], y), y, color='r')
        

    # 点群データ(ROI)
    """
    ax = plt.subplot(132)
    plt.imshow(view_image, cmap='gray')
    for roiname, item in fit_items.items():
        plt.scatter(item['raw'][:,0], item['raw'][:,1], fc='None', edgecolors='blue')
    """

    # 点群データ
    """
    ax = plt.subplot(133)
    plt.imshow(view_image, cmap='gray')
    for i, v in data.items():
        plt.scatter(v[:,0] + area[i,0] - area[i,2] / 2, v[:,1] + area[i,1] - area[i,3] / 2, fc='None', edgecolors='blue')

    """
    plt.show()

    # JSONファイルとして集計
    with open( basepath + filename + "_roi.json", 'w') as json_roi_file:
        json.dump(data_save_roi, json_roi_file, indent=4)

    # テンプレート画像として保存
    cv2.imwrite( basepath + "tp_" + filename + ".bmp", view_image_trim)

    # 部分領域もテンプレート画像として保存
    for ck, cv in data_save_roi.items():
        if 'region' in cv:
            if 'enable' not in context[ck] or context[ck]['enable'] == False:
                continue

            
            region = cv['region']

            trim_image = view_image_trim[region[1]:region[3], region[0]:region[2]]
            trim_image = cv2.GaussianBlur(trim_image, (5,5), 2.0)
            trim_image_flatten = trim_image.flatten()
            # ノイズ除去してデータを保存できるか試してみる
            H_csr = lil_matrix((trim_image_flatten.shape[0], trim_image_flatten.shape[0]))
            for i in range(0, trim_image_flatten.shape[0]):
                H_csr[i, i] = 1

            alpha_val = 0.0001
            if 'alpha' in context[ck]:
                alpha_val = context[ck]['alpha']
            model = Lasso(alpha=alpha_val).fit(H_csr, trim_image_flatten)
            image_raw_conv = model.coef_

            # 閾値処理
            hist_lasso = np.histogram( image_raw_conv, range=(np.min(image_raw_conv),np.max(image_raw_conv)), bins=256)
            hist_lasso_max = hist_lasso[1][np.argmax(hist_lasso[0])]
            image_raw_conv_2 = np.where(image_raw_conv < hist_lasso_max, image_raw_conv, hist_lasso_max)

            # 差分処理
            minval = np.min(image_raw_conv_2)
            image_raw_conv_2 = image_raw_conv_2 - minval
            hist_lasso_max = hist_lasso_max - minval
            image_raw_conv_2 = np.where(image_raw_conv_2 < hist_lasso_max, image_raw_conv_2, 255)
            image_raw_conv_2 = np.array(image_raw_conv_2, dtype=np.uint8)


            #plt.imshow(image_raw_conv_2.reshape(trim_image.shape[0], trim_image.shape[1]), cmap='gray')
            #plt.show()
            cv2.imwrite( basepath + cv['tppath'], image_raw_conv_2.reshape(trim_image.shape[0], trim_image.shape[1]))
            print("Exported > " + str(cv['tppath']))

