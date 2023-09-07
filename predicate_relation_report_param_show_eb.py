import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


def draw(xt, yt, heatlist, xlb='Number of iteration rounds', ylb='Number of network layers', flag_arr='precision_rel_bert'):
    plt.imshow(heatlist, cmap='Blues_r')  # bone
    # 显示右边的栏
    plt.colorbar()
    plt.xticks(range(len(xt)), xt)
    plt.yticks(range(len(yt)), yt)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.show()


if __name__ == '__main__':

    text_file = open('logs/relation_paramrelations_data_all.txt', mode='r', encoding='utf-8')  # 打开文件，文件存在则打开，不存在则创建后再打开

    json_dta, emb, sub_emb = [], [], []
    for dtas in text_file:
        j_dta = json.loads(dtas)
        json_dta.append(j_dta)

    flag_arr = 'f1_text_gcn'  # 'precision_text_gcn', 'recall_text_gcn',  'f1_text_gcn'

    heatlist = []
    for emb in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  #
        sub_lis = []
        for ep in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  #
            for j_d in json_dta:
                if j_d['epoch'] == ep and j_d['ly'] == emb:
                    sub_lis.append(j_d[flag_arr])
                    break
        heatlist.append(sub_lis)

    # 定义横纵坐标
    xt = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    yt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    draw(xt, yt, np.array(heatlist), flag_arr=flag_arr)
