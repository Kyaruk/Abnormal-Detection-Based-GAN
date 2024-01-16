# import time
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# import random
#
# # 准备数据
# x_data = ["Layer1", "Layer2", "Layer3", "Layer4", "OurModel",
#           "OC-SVM", "DAGMM", "AnoGAN", "BiGAN"]
# y_data = [0.9474, 0.9236, 0.9472, 0.9351, 0.9483, 0.7457, 0.9297, 0.8786, 0.9200]
# y_data = [0.9677, 0.9498, 0.9500, 0.9483, 0.9539, 0.8523, 0.9442, 0.8297, 0.9582]
# y_data = [0.9409, 0.9474, 0.9468, 0.9451, 0.9473, 0.7954, 0.9369, 0.8865, 0.9372]
#
# # 正确显示中文和负号
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
#
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
#
# # 画图，plt.bar()可以画柱状图
# for i in range(len(x_data)):
#     plt.bar(x_data[i], y_data[i])
# # 设置图片名称
# plt.title("F1分数")
# # 设置x轴标签名
# plt.xlabel("模型")
#
# for i, j in zip(x_data, y_data):
#     plt.text(i, j + 0.01, "{:.2%}".format(j), ha="center", va="bottom", fontsize=10)
#     # time.sleep(1000)
#
# plt.ylim(0.7, 1)
#
# # 设置y轴标签名
# plt.ylabel("F1分数")
# # 显示
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

size = 4
# 返回size个0-1的随机数
# precision
a = [0.9499, 0.9731, 0.9549, 0.8741]
b = [0.9474, 0.9667, 0.9611, 0.8766]
c = [0.9794, 0.9821, 0.9739, 0.8863]

# # recall
# a = [0.9483, 0.9677, 0.8995, 0.9019, 0.9293]
# b = [0.9454, 0.9277, 0.9288, 0.9355, 0.9343]
# c = [0.9677, 0.9498, 0.9500, 0.9483, 0.9539]

# # F1
# a = [0.9473, 0.9661, 0.9468, 0.9195, 0.9455]
# b = [0.9198, 0.9445, 0.9166, 0.9072, 0.9220]
# c = [0.9499, 0.9474, 0.9468, 0.9451, 0.9473]

# x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
x = np.arange(size)

# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.9, 3
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)

plt.ylim(0.86, 1)

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.title("生成器和判别器消融实验结果图", fontsize=16)
plt.ylabel("F1分数", fontsize=14)
plt.xlabel("数据集", fontsize=14)

plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# 画柱状图
plt.bar(x, a, width=width, label="生成器G")
plt.bar(x + width, b, width=width, label="判别器D")
plt.bar(x + 2 * width, c, width=width, label="综合")
# 显示图例
plt.legend(fontsize=12)
plt.yticks(fontsize=14)

# 功能1
x_labels = ["KDD CUP 99", "NSL-KDD", "ISCXTor2016", "UNSW-NB15"]
# 用第1组...替换横坐标x的值
plt.xticks(x + width, x_labels, fontsize=14)

# 功能2
flag = 0
for i, j in zip(x, a):
    if flag == 0:
        plt.text(i, j + 0.002, "{:.2%}".format(j), ha="center", va="bottom", fontsize=9)
    else:
        plt.text(i, j, "{:.2%}".format(j), ha="center", va="bottom", fontsize=9)
    flag += 1

flag = 0
for i, j in zip(x + width, b):
    if flag == 0 or 1:
        plt.text(i, j, "{:.2%}".format(j), ha="center", va="bottom", fontsize=9)
    else:
        plt.text(i, j + 0.002, "{:.2%}".format(j), ha="center", va="bottom", fontsize=9)
    flag += 1

for i, j in zip(x + 2 * width, c):
    plt.text(i, j , "{:.2%}".format(j), ha="center", va="bottom", fontsize=9)

# 显示柱状图
plt.show()
