import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

basepath = '..\\..\image\\practice\\'

imagedir = basepath + 'sample_1_org.bmp'
view_image = cv2.imread(imagedir)

with open(basepath + 'round_spline.json', 'r') as f:
    tp_info = json.load(f)

export_filedir = basepath + tp_info['export_filedir']

txt = np.transpose(np.loadtxt(export_filedir, delimiter = ','))
r_theta1 = txt[0, :]
r_theta2 = txt[1, :]
r_cenx = txt[2, :] * 8.0 + 4.0
r_ceny = txt[3, :] * 8.0 + 4.0
r_rad = txt[4, :] * 8.0 + 4.0
pass




plt.figure()
ax = plt.axes()

plt.imshow(view_image, cmap='gray')
for i in range(0, len(r_theta1)):
    ax.add_patch(patches.Arc(xy=(r_cenx[i],r_ceny[i]), width=r_rad[i]*2, height=r_rad[i]*2, fc='None', ec='red', theta1=r_theta1[i], theta2=r_theta2[i]))



plt.show()




