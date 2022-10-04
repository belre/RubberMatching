
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from scipy import interpolate

basepath = '..\\..\image\\practice\\'

with open(basepath + 'round_spline.json', 'r') as f:
    tp_info = json.load(f)

ref_image = cv2.imread(basepath + tp_info['reference']['viewed_image'])
export_filedir = basepath + tp_info['export_filedir']

plt.figure()
ax = plt.axes()

with open(export_filedir, 'w') as f_resout:
    #f_resout.write('theta1,theta2,cenx,ceny,radius\n')
    
    for item in tp_info['sample']:
        plist = np.loadtxt(basepath + '\\'+ item + '.csv', delimiter=',')

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

        # ベクトル全体加算
        # ここで指定された方向をy軸に再設定する
        total_v = np.array( (np.average(plist_conv[:, 0]), np.average(plist_conv[:, 1])) )
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

        plt.scatter(plist_conv[:,0] + center_round[0], plist_conv[:,1] + center_round[1], c='None', edgecolors='blue')
        ax.add_patch(patches.Arc(xy=(center_round), width=radius*2, height=radius*2, fc='None', ec='red', theta1=np.rad2deg(dec_to_y1), theta2=np.rad2deg(dec_to_y2)))


        f_resout.write(format(np.rad2deg(dec_to_y1), '.4f') + "," + format(np.rad2deg(dec_to_y2), '.4f') + ",")
        f_resout.write(format(center_round[0], '.4f') + "," + format(center_round[1], '.4f') + "," + format(radius, '.4f') + "\n")






plt.imshow(ref_image, cmap='gray')

plt.show()
