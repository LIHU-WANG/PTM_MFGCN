import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

warnings.filterwarnings("ignore")


def draw(xt, yt, heatlist, xlb='Number of iteration rounds', ylb='Learning rate', flag_arr='precision_rel_bert'):
    plt.imshow(heatlist, cmap='Blues_r')  # bone
    # 显示右边的栏
    plt.colorbar()
    plt.xticks(range(len(xt)), xt)
    plt.yticks(range(len(yt)), yt)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.show()


if __name__ == '__main__':

    text_file = open('logs/relation_param_rt_2_relations_data_all.txt', mode='r', encoding='utf-8')  # 打开文件，文件存在则打开，不存在则创建后再打开

    json_dta, emb, sub_emb = [], [], []
    for dtas in text_file:
        j_dta = json.loads(dtas)
        json_dta.append(j_dta)

    flag_arr = 'recall_text_gcn'  # 'precision_text_gcn', 'recall_text_gcn',  'f1_text_gcn'

    heatlist = []
    for emb in [0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]:  #
        sub_lis = []
        for ep in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  #
            for j_d in json_dta:
                if j_d['epoch'] == ep and j_d['rt'] == emb:
                    sub_lis.append(j_d[flag_arr])
                    break
        heatlist.append(sub_lis)

    # 定义横纵坐标
    xt = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    yt = [0.000000005, 0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    draw(xt, yt, np.array(heatlist), flag_arr=flag_arr)
