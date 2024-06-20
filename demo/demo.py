"""
通过可视化展示基于二进制哈希码进行图像检索的结果。
具体来说，它从保存的哈希码和标签中加载数据，选择一些查询图像，并展示检索到的最近邻图像以及它们与查询图像的相似性。
"""

import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


# 从指定目录加载训练集和测试集的二进制哈希码和标签。
# 从图像路径文件中读取训练集和测试集的图像路径。
data_dir = "/data/xyj/2021/DeepHash-pytorch/save/DPSH/imagenet_64bits_0.8824931967229359/"
trn_binary = np.load(data_dir + "trn_binary.npy")
trn_label = np.load(data_dir + "trn_label.npy")
tst_binary = np.load(data_dir + "tst_binary.npy")
tst_label = np.load(data_dir + "tst_label.npy")

img_dir = "/dataset/imagenet/"
with open("./../data/imagenet/database.txt", "r") as f:
    # 读取database每张图片的路径
    trn_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]
with open("./../data/imagenet/test.txt", "r") as f:
    # 读取test_data每张图片的路径
    tst_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]


# 设置可视化参数：m 表示查询图像的数量，n 表示每个查询图像的最近邻数量。
# 创建一个画布，并设置图像的大小和分辨率。
# 随机选择 m 个测试样本作为查询图像。
m = 5
n = 8
plt.figure(figsize=(40, 20), dpi=50)
font_size = 30
"""
tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]
这行代码的目的是从测试集中随机选择 m 张图像作为查询图像。具体步骤如下：
range(tst_binary.shape[0]):
    tst_binary 是测试集中所有图像的二进制哈希码矩阵。
    tst_binary.shape[0] 获取测试集中的图像数量。
    range(tst_binary.shape[0]) 生成一个从 0 到 tst_binary.shape[0] - 1 的序列，表示测试集中所有图像的索引。
np.random.permutation(range(tst_binary.shape[0])):
    np.random.permutation 将输入序列进行随机排列，返回一个新的乱序排列的数组。
    通过对测试集图像索引进行随机排列，可以确保从测试集中随机选择图像。
[0: m]:
    [0: m] 从随机排列的索引数组中选择前 m 个索引。
    这些索引对应于测试集中随机选择的 m 张图像
"""
tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]

# 可视化查询和检索结果
for row, query_index in enumerate(tst_select_index):

    # 计算查询图像与训练集中所有图像标签的点积，得到 gnd，表示相关性。
    # 计算查询图像与训练集中所有图像的汉明距离。
    # 根据汉明距离排序，取前 n 个最近邻的索引。
    query_binary = tst_binary[query_index]
    query_label = tst_label[query_index]
    # 计算测试集和检索是否相似
    gnd = (np.dot(query_label, trn_label.transpose()) > 0).astype(np.float32)
    # 通过哈希码计算汉明距离
    hamm = CalcHammingDist(query_binary, trn_binary)
    # 计算最近的n个距离的索引
    ind = np.argsort(hamm)[:n]
    # 返回结果的真值
    t_gnd = gnd[ind]
    # 返回结果的汉明距离
    q_hamm = hamm[ind].astype(int)

    q_img_path = tst_img_path[query_index]
    return_img_list = np.array(trn_img_path)[ind].tolist()


    # 显示查询图像。
    # 显示检索结果图像，并根据其与查询图像的相关性，标注 √ 或 ×，并用不同颜色框表示相关性。
    plt.subplot(m, n + 1, row * (n+1) + 1)

    img = Image.open(q_img_path).convert('RGB').resize((128, 128))
    plt.imshow(img)
    plt.axis('off')
    plt.text(5, 145, 'query image', size=font_size)

    for index, img_path in enumerate(return_img_list):
        # plt.subplot(1, n + 1, index + 2)
        plt.subplot(m, n + 1, row * (n+1) + index + 2)
        img = Image.open(img_path).convert('RGB').resize((120, 120))
        if t_gnd[index]:
            plt.text(60, 145, '√', size=font_size)
            img = ImageOps.expand(img, 4, fill=(0, 0, 255))
        else:
            plt.text(60, 145, '×', size=font_size)
            img = ImageOps.expand(img, 4, fill=(255, 0, 0))
        plt.axis('off')
        plt.imshow(img)
plt.savefig("demo.png")
plt.show()
