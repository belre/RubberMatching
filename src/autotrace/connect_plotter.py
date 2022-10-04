import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


basepath = '..\\..\\image\\autotrace\\'

dist_th = 8*10


with open(basepath + 'connect.json', 'r') as f:
    tp_info = json.load(f)

imagedir = basepath + tp_info['reference']['viewed_image']
view_image = cv2.imread(imagedir)


export_filedir = basepath + tp_info['export_filedir']



for filename, context in tp_info['sample'].items():
    data = np.loadtxt(basepath + filename + ".csv", delimiter = ',')

    plt.figure()
    ax = plt.axes()

    plt.title(filename)
    plt.scatter(data[:,0], data[:,1])
    plt.plot(data[:,0], data[:,1])
    for t in data:
        ax.add_patch(patches.Rectangle(xy=(t[0]-t[2]/2, t[1]-t[3]/2), width=t[2], height=t[3], fill=False, ec='red'))



    plt.imshow(view_image, cmap='gray')


    plt.show()




