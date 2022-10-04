
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# 時間計測クラス
class measure:
    start = time.time()
    isdisplay = True

    def __init__(self):
        self.start = time.time()

    def measure_time(self):
        elapsed_time = time.time() - self.start
        if self.isdisplay == True:
            print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        self.start = time.time()

# 位置によって最小の値になるような画像情報を生成する
def fig_map(shape, ref_points, priority=True, **kwargs):
    m_obj=measure()

    width = shape[0]
    height = shape[1]

    eval_func = lambda r, x : np.linalg.norm(r - x, ord=2, axis=2)
    if 'eval_func' in kwargs:
        if type(kwargs['eval_func']) is type(fig_map):
            eval_func = kwargs['eval_func']

    # 距離最小値となる画素番号を抽出
    coord_mat = np.array([[ [i, j] for i in range(0, width)] for j in range(0, height)])
    minargs_mat = np.zeros((height, width), dtype='int64')
    min_mat = np.ones((height, width)) * sys.float_info.max

    # index=0の点を最初に追加する
    # 追加した情報は、priority==Trueの場合-1に、それ以外は>=0に置き換えるようにする.
    ref_points_add0 = np.vstack( (np.array((0, 0)), ref_points))
    
    # onesは前もって用意する
    ones = np.ones((height, width))

    # データ点に対して計算
    for k in range(1, ref_points_add0.shape[0]):
        # ref番号の添え字の配列を生成
        k_mat = ones * k

        # 評価値を現在の座標点と参照点から演算
        tmp_mat = eval_func(ref_points_add0[k], coord_mat[:, :])

        # 符号を計算.
        # priorityがTrueの場合は0も+とする
        sig_mat = None 
        if priority == True:
            sig_mat = (-2 * (np.signbit(min_mat - tmp_mat)  - 1/2)).astype(np.int64)
        else:
            sig_mat = np.sign(min_mat - tmp_mat)

        # ±のみを考慮した値の算出 (※同じ場合は1/2が出力されるので、補正項を足す)
        sig_mat_pm = (sig_mat + 1) / 2

        # ラベリング画像
        if priority == True:
            minargs_mat = ((1 - sig_mat_pm) * minargs_mat + sig_mat_pm * k_mat)
        else:
            minargs_mat_tmp = ((1 - sig_mat_pm) * minargs_mat + sig_mat_pm * k_mat)
            minargs_mat = minargs_mat_tmp - ((minargs_mat + k_mat)/2) * (1-np.abs(sig_mat))

        minargs_mat = minargs_mat.astype(np.int64)

        # 距離画像
        min_mat = (1 - sig_mat_pm) * min_mat + sig_mat_pm * tmp_mat

    m_obj.measure_time()
    
    return min_mat, (minargs_mat-1)

# 幅, 高さ
width = 500
height = 500

# 参照点情報
random_num = 100
ref_points = np.zeros((random_num, 2))
for i in range(0, random_num):
    ref_points[i] = np.array((width * np.random.rand(), height * np.random.rand()))

# 二等分線図
plt.figure()
dist_mat, label_mat = fig_map((width, height), ref_points, True)
plt.subplot(121)
plt.imshow(label_mat, cmap='plasma')
plt.subplot(122)
plt.imshow(dist_mat, cmap='gray')
plt.show()

# L1ノルムで分割
plt.figure()
dist_mat, label_mat = fig_map((width, height), ref_points, True, eval_func=lambda r, x : np.linalg.norm(r - x, ord=1, axis=2))
plt.subplot(121)
plt.imshow(label_mat, cmap='plasma')
plt.subplot(122)
plt.imshow(dist_mat, cmap='gray')
plt.show()

# マフィンティン・ポテンシャル
def muffin(r, x):
    thresh = 20.0
    dist = np.linalg.norm(r - x, axis=2)
    dist = np.sign(dist - thresh * np.ones((x.shape[0], x.shape[1])))
    return dist

plt.figure()
dist_mat, label_mat = fig_map((width, height), ref_points, False, eval_func=muffin)
plt.subplot(121)
plt.imshow(label_mat)
plt.subplot(122)
plt.imshow(dist_mat, cmap='gray')
plt.show()
