from K_means import *
from Fuzzy_C_means import *
import matplotlib.pyplot as plt
import numpy as np

def load_data_set(fileName):
    """加载数据集"""
    dataSet = []  # 初始化一个空列表
    fr = open(fileName)
    for line in fr.readlines():
        # 按tab分割字段，将每行元素分割为list的元素
        curLine = line.strip().split('\t')
        # 其中map(float, curLine)表示把列表的每个值用float函数转成float型，并返回迭代器
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet

# 测试K-means
# test = np.array(load_data_set('testSet.txt'))
# C, label = K_Means(test, k=4, error=0.001, T=100, random_state=2020)
#
#
# label_map = {0: 'r', 1:'b', 2:'g', 3:'y'}
# label_color = list(map(lambda x: label_map[x], label))
# plt.scatter(x=test[:, 0], y=test[:, 1], c=label_color)
# plt.show()

# 测试Fuzzy C-means
test = np.array(load_data_set('testSet.txt'))
C, U = C_Means(test, k=4, error=0.001, T=100, random_state=2020)
label = np.argmax(U, axis=1)

label_map = {0: 'r', 1:'b', 2:'g', 3:'y'}
label_color = list(map(lambda x: label_map[x], label))
plt.scatter(x=test[:, 0], y=test[:, 1], c=label_color)
plt.show()

