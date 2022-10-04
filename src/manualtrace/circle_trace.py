
# 円領域を画像処理ソフトで測定して
# 軌道を抽出する


import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

basepath = 'F:\\LocalData\\Tokuyama\\manualtrace\\'


with open(basepath + 'circle_trace.json', 'r') as f:
    tp_info = json.load(f)



data_width = 90

for filename, context in tp_info['sample'].items():
    imgfile = basepath + filename + ".bmp"
    image = cv2.imread(imgfile)


    coord_fix = context['fix_coord']
    region = context['region']
    rot_deg = -context['rot_degree'] 
    meas_deg = context['meas_degree'] * np.pi / 180

    # 中心座標
    center0 = np.array((region[0]+region[2]/2, region[1]+region[3]/2))

    # 楕円パラメータ
    len_ax = region[2] / 2
    len_ay = region[3] / 2

    # 画像回転
    rot_trans = cv2.getRotationMatrix2D( (image.shape[1]/2,image.shape[0]/2), rot_deg, 1.0)
    rot_image =  cv2.warpAffine(image, rot_trans, (image.shape[1], image.shape[0]))


    fix = plt.figure(figsize=(16, 9))
    ax = plt.subplot(121)
    plt.imshow( rot_image )
    plt.scatter(center0[0], center0[1])

    # 開始点, 終了点
    coord_fix_s = np.array(coord_fix[0]) - center0
    angle_s = np.arctan2(coord_fix_s[1], coord_fix_s[0]) 
    coord_fix_e = np.empty(2)
    coord_fix_e[0] = (coord_fix_s[0] * np.cos(meas_deg) / len_ax - coord_fix_s[1] * np.sin(meas_deg) / len_ay) * len_ax
    coord_fix_e[1] = (coord_fix_s[0] * np.sin(meas_deg) / len_ax + coord_fix_s[1] * np.cos(meas_deg) / len_ay) * len_ay
    angle_e = np.arctan2(coord_fix_e[1], coord_fix_e[0])
    
    num_division = int((angle_e - angle_s) * len_ay + 0.5) 

    # 空間補正後のデータを保持
    cor_image = np.zeros((num_division, data_width))


    for j in range(0, num_division):
        angle_i = angle_s + j * (angle_e - angle_s) / num_division
        coord = np.array(( center0[0] + region[2] * np.cos(angle_i) / 2, center0[1] + region[3] * np.sin(angle_i) / 2 ))

        normal_vec = (region[2] * np.cos(angle_i) / 2, region[3] * np.sin(angle_i) / 2)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)

        for i in range(0, data_width):
            coord_t = coord + (normal_vec * (i - data_width / 2)) 
            coord_t_int = coord_t.astype('int64')
            coord_t_dev = coord_t - coord_t_int
            
            # bilinear
            cor_image[j, i] = (1 - coord_t_dev[0]) * (1 - coord_t_dev[1]) * rot_image[coord_t_int[1], coord_t_int[0], 0]
            cor_image[j, i] += coord_t_dev[0] * (1 - coord_t_dev[1]) * rot_image[coord_t_int[1], coord_t_int[0]+1, 0]
            cor_image[j, i] += (1 - coord_t_dev[0]) * coord_t_dev[1] * rot_image[coord_t_int[1]+1, coord_t_int[0], 0]
            cor_image[j, i] += coord_t_dev[0] * coord_t_dev[1] * rot_image[coord_t_int[1]+1, coord_t_int[0]+1, 0]
            pass


    for i in range(0, num_division):
        angle_i = angle_s + i * (angle_e - angle_s) / num_division
        plt.scatter( center0[0] + region[2] * np.cos(angle_i) / 2, center0[1] + region[3] * np.sin(angle_i) / 2, edgecolors='g', facecolor='g', alpha=0.5)

    ax.add_patch(patches.Arc(xy=center0, width=region[2], height=region[3], ec='r', Fill=False, theta1=angle_s * 180 / np.pi, theta2=angle_e * 180 / np.pi))

    ax = plt.subplot(122)
    plt.imshow(cor_image, cmap='gray')
    #plt.show()

    cv2.imwrite(basepath + '\\export\\' + filename + "_ext" + '.bmp', cor_image)


    pass
