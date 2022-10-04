import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


basepath = '..\\..\\image\\autotrace\\'

dist_th = 8*10


with open(basepath + 'plotter.json', 'r') as f:
    tp_info = json.load(f)

imagedir = basepath + tp_info['reference']['viewed_image']
view_image = cv2.imread(imagedir)


export_filedir = basepath + tp_info['export_filedir']

plt.figure()
ax = plt.axes()

for filename, context in tp_info['sample'].items():
    data = np.loadtxt(basepath + filename + ".csv", delimiter = ',')

    # dataを距離が近しいものをまとめる
    c_index = 0
    labelar = [-1] * len(data)
    c_data = {}
    for i in range(0, len(data)):
        if labelar[i] != -1:
            continue

        labelar[i] = c_index
        vec1= np.array((data[i,0], data[i,1]))
   
        c_data[c_index] = np.vstack( (np.empty( (0,2) ), vec1) )

        for j in range(i+1, len(data)):
            if labelar[j] != -1:
                continue

            vec2 = np.array((data[j,0], data[j,1]))
            norm = np.linalg.norm(vec2 - vec1)
            if np.linalg.norm(vec2 - vec1) < dist_th :
                labelar[j] = c_index
                c_data[c_index] = np.vstack( (c_data[c_index], vec2))

        c_index = c_index + 1

    # まとめた結果で代表値を算出
    plot_data = np.zeros((len(c_data.keys()), 2))
    for i, veclist in c_data.items():
        plot_data[i] = np.array( (np.average(veclist[:,0]), np.average(veclist[:,1])) )

    # 本来はここで一筆書きになるように
    # 最小化するための解析があると便利
    # とりあえずパターンは1パターンで検討なので、手動で並び替える
    with open(basepath + '\\export\\' + filename + '_convex.txt', 'w') as f:
        for k in plot_data:
            f.write( format(k[0], '.4f') + ',' + format( k[1], '.4f') + '\n')
                

    plt.scatter(plot_data[:,0], plot_data[:,1])
    plt.plot(plot_data[:,0], plot_data[:,1])




plt.imshow(view_image, cmap='gray')


plt.show()




