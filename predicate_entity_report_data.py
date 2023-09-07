import matplotlib.pyplot as plt
import numpy as np


# BILSTM          = [80.37, 88.98, 84.17]
# BERT_BILSTM_CRF = [93.49, 94.87, 94.09]
# TextCNN         = [88.46, 89.66, 88.93]
# BERT_CNN        = [93.60, 94.81, 94.19]
# FastText        = [78.03, 38.60, 42.52]
# PTM_MFGCN       = [94.65, 97.28, 95.94]

BILSTM          = [79.50, 77.90, 77.46]
BERT_BILSTM_CRF = [81.56, 78.67, 78.06]
TextCNN         = [55.98, 30.90, 23.33]
BERT_CNN        = [68.51, 66.14, 61.60]
FastText        = [71.56, 71.43, 70.55]
PTM_MFGCN       = [85.18, 84.33, 84.60]

name_arr_x = ['${P}$', '${R}$', '${F1}$']
name_step = 3
move_step = 0.54
x = np.arange(name_step)
total_width, n = 0.38, 2.5  # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 将字体设置为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 显示坐标轴负数
plt.bar(x + 1 * width, BILSTM, width=width, label='BILSTM', color='#e6edf4')
plt.bar(x + 2 * width, BERT_BILSTM_CRF, width=width, label='BERT-BILSTM-CRF', color='#9cb7d4')
plt.bar(x + 3 * width, TextCNN, width=width, label='TextCNN', color='#5281b4')
plt.bar(x + 4 * width, BERT_CNN, width=width, label='BERT-CNN', color='#084b94')
plt.bar(x + 5 * width, FastText, width=width, label='FastText', color='#063568')
plt.bar(x + 6 * width, PTM_MFGCN, width=width, label='PTM-MFGCN', color='#02172c')

for a, b in zip(x + 1 * width, BILSTM):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)
for a, b in zip(x + 2 * width, BERT_BILSTM_CRF):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)
for a, b in zip(x + 3 * width, TextCNN):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)
for a, b in zip(x + 4 * width, BERT_CNN):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)
for a, b in zip(x + 5 * width, FastText):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)
for a, b in zip(x + 6 * width, PTM_MFGCN):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=9)

plt.ylabel('Index value')
plt.xticks([i + move_step for i in x], name_arr_x)
plt.xticks(rotation=0)  # 45为旋转的角度
plt.legend()
plt.show()
