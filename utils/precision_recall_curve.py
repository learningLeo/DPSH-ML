"""
从多个方法生成的 PR 曲线数据文件中读取数据，
然后绘制这些方法的 Precision-Recall (PR) 曲线，
并将其保存为图像文件。
这些 PR 曲线用于评估和比较不同图像检索方法的性能。
"""

import matplotlib.pyplot as plt
import json
import os
# 设置 matplotlib 参数，以支持中文字体和正确显示负号。
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个字典 pr_data，其中包含多个方法的 PR 数据文件路径。
# Precision Recall Curve data
# CNNF-CIFAR
pr_data = {
    "DPSH_CNNF_cifar_12": "../log/CNNF/DPSH_CNNF_cifar10-1_12.json",
    "DPSH_CNNF_cifar_24": "../log/CNNF/DPSH_CNNF_cifar10-1_24.json",
    "DPSH_CNNF_cifar_32": "../log/CNNF/DPSH_CNNF_cifar10-1_32.json",
    "DPSH_CNNF_cifar_48": "../log/CNNF/DPSH_CNNF_cifar10-1_48.json",
}

# CNNF-NUSWIDE
# pr_data = {
#     "DPSH_CNNF_nuswide_12": "../log/CNNF/DPSH_CNNF_nuswide_21_12.json",
#     "DPSH_CNNF_nuswide_24": "../log/CNNF/DPSH_CNNF_nuswide_21_24.json",
#     "DPSH_CNNF_nuswide_32": "../log/CNNF/DPSH_CNNF_nuswide_21_32.json",
#     "DPSH_CNNF_nuswide_48": "../log/CNNF/DPSH_CNNF_nuswide_21_48.json",
# }


# ResNet-CIFAR
# pr_data = {
#     "DPSH_ResNet_cifar_12": "../log/ResNet/DPSH_ResNet_cifar10-1_12.json",
#     "DPSH_ResNet_cifar_24": "../log/ResNet/DPSH_ResNet_cifar10-1_24.json",
#     "DPSH_ResNet_cifar_32": "../log/ResNet/DPSH_ResNet_cifar10-1_32.json",
#     "DPSH_ResNet_cifar_48": "../log/ResNet/DPSH_ResNet_cifar10-1_48.json",
# }


# ResNet-NUSWIDE
# pr_data = {
#     "DPSH_ResNet_nuswide_12": "../log/ResNet/DPSH_ResNet_nuswide_21_12.json",
#     "DPSH_ResNet_nuswide_24": "../log/ResNet/DPSH_ResNet_nuswide_21_24.json",
#     "DPSH_ResNet_nuswide_32": "../log/ResNet/DPSH_ResNet_nuswide_21_32.json",
#     "DPSH_ResNet_nuswide_48": "../log/ResNet/DPSH_ResNet_nuswide_21_48.json",
# }

# CIFAR-CNNF-RESNET
# pr_data = {
#     "DPSH_CNNF_cifar_24": "../log/CNNF/DPSH_CNNF_cifar10-1_24.json",
#     "DPSH_CNNF_cifar_48": "../log/CNNF/DPSH_CNNF_cifar10-1_48.json",
#     "DPSH_ResNet_cifar_24": "../log/ResNet/DPSH_ResNet_cifar10-1_24.json",
#     "DPSH_ResNet_cifar_48": "../log/ResNet/DPSH_ResNet_cifar10-1_48.json",
# }

# # NUSWIDE-CNNF-RESNET
# pr_data = {
#     "DPSH_CNNF_nuswide_24": "../log/CNNF/DPSH_CNNF_nuswide_21_24.json",
#     "DPSH_CNNF_nuswide_48": "../log/CNNF/DPSH_CNNF_nuswide_21_48.json",
#     "DPSH_ResNet_nuswide_24": "../log/ResNet/DPSH_ResNet_nuswide_21_24.json",
#     "DPSH_ResNet_nuswide_48": "../log/ResNet/DPSH_ResNet_nuswide_21_48.json",
# }


# 设置 N 为 150，表示绘制曲线时最多取前 150 个数据点。
# 遍历 pr_data 字典，读取每个方法的 PR 数据文件，并将其加载为 JSON 数据。
N = 150
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# 定义一个字符串 markers，包含不同的 marker 符号。
# 为每个方法分配一个 marker 符号，保存在字典 method2marker 中。
# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1


plt.figure(figsize=(15, 5))
plt.subplot(131)

"""
绘制 Precision-Recall 曲线
    创建一个 15x5 英寸的画布。
    在第一个子图（左图）中绘制 Precision-Recall 曲线。
    读取每个方法的精度 (P) 和召回率 (R) 数据。
    使用 plt.plot 绘制曲线，设置线型和 marker。
    设置网格、坐标轴范围和标签。
    添加图例。
"""
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)

"""
绘制 Recall 曲线
在第二个子图（中图）中绘制 Recall 曲线。
读取前 N 个召回率 (R) 和检索样本数 (index) 数据。
使用 plt.plot 绘制曲线。
设置网格、坐标轴范围和标签。
添加图例。
"""
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

"""
绘制 Precision 曲线
在第三个子图（右图）中绘制 Precision 曲线。
读取前 N 个精度 (P) 和检索样本数 (index) 数据。
使用 plt.plot 绘制曲线。
设置网格、坐标轴范围和标签。
添加图例。
"""
plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()

plt.savefig("pr-CNNF-CIFAR.png")
# plt.savefig("pr-CNNF-NUSWIDE.png")
# plt.savefig("pr-ResNet-CIFAR.png")
# plt.savefig("pr-ResNet-NUSWIDE.png")

# plt.savefig("pr-CIFAR-CNNF-ResNet.png")
# plt.savefig("pr-nuswide-CNNF-ResNet.png")
plt.show()
