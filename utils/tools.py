"""
utils.tools.py 文件提供了用于配置数据集、加载数据、计算结果和评估性能的工具函数。
其核心功能包括:
    配置数据集路径
    图像预处理
    数据加载
    计算二进制哈希码
    计算汉明距离和平均准确率 (MAP)
    在验证集上评估模型性能
通过这些工具函数，整个 DPSH 模型的训练和评估过程得以顺利进行。
"""

import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json


# ------------------------------------------------------------------------------------------------------------------------

# 设置config
"""
配置数据集函数：config_dataset
该函数根据数据集名称配置相应的参数，如类别数量 (n_class)、检索数量 (topK)、数据路径 (data_path)，并返回更新后的配置字典。
    
    topk：
        用于图像检索任务中计算平均准确率（MAP）时要考虑的前 K 个检索结果，-1 表示不限制检索数量
    n_class：
        数据集中的类别数量，不同的数据集有不同的类别数量
    data_path：
        根据不同的数据集，设置相应的数据集路径
    data：
        数据集的详细配置，包括训练集、数据库和测试集的路径及批量大小。
        一个字典，包含 train_set、database 和 test 三个子项，每个子项中包含 list_path（数据集路径）和 batch_size（批量大小）
"""
def config_dataset(config):
    if "cifar" in config["dataset"]:
        # config["topK"] = -1
        config["topK"] = 5000
        config["n_class"] = 10
    elif config['dataset'] == 'numswide_10':
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38


    config["data_path"] = "/root/autodl-tmp/zhjproject/DPSH/dataset/" + config["dataset"] + "/"
    
    config["data"] = {
        "train_set": {"list_path": "/root/autodl-tmp/zhjproject/DPSH/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "/root/autodl-tmp/zhjproject/DPSH/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "/root/autodl-tmp/zhjproject/DPSH/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    
    return config

# ------------------------------------------------------------------------------------------------------------------------

# 获取CIFAR 查询集 训练集 数据集库
"""
自定义CIFAR-10类：MyCIFAR10
    继承自 torchvision.datasets.CIFAR10，重载了 __getitem__ 方法，使其返回图像、目标和索引。
"""
class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


"""
需要生成cifar-10数据集的train text database 的index
    generate_cifar_index(config)根据cifar数据类型生成
"""
def generate_cifar_index(config):
    # 1. 设置批量大小和数据集大小

    # 从 config 中获取批量大小。
    # 默认情况下，每类train_size 为 500，test_size 为 100。
    train_size = 500
    test_size = 100

    # 如果数据集是 "cifar10-1"，则 train_size 为 2000，test_size 为 400。
    if config["dataset"] == "cifar10-1":
        train_size = 2000  # 每类2000张训练
        test_size = 400    # 每类400张测试
    # end

    # 如果数据集是 "cifar10-2"，则 train_size 为 5000，test_size 为 1000。
    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    # 2. 定义数据预处理步骤
    # 使用 transforms.Compose 组合了一系列的图像预处理步骤，包括调整图像大小、将图像转换为张量、归一化图像等。
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/root/autodl-tmp/zhjproject/DPSH/dataset/cifar/'
    
    # 3.加载数据集
    # 使用自定义的 MyCIFAR10 类加载训练集、测试集和数据库集数据。
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    # 4. 合并数据
    # 将训练集和测试集的数据合并，得到完整的数据 X 和标签 L。
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    # 5. 根据标签划分数据
    # 按照标签，将数据随机打乱，并划分为训练集、测试集和数据库集。
    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False


    # 6. 根据数据集类型调整数据库集
    # 如果数据集类型为 "cifar10-1"，则将训练集数据并入数据库集中。
    # 如果数据集类型为 "cifar10-2"，则数据库集仅包含训练集数据。
    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:4000, train:20000, database:56000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index
    # 保存 index
    np.savetxt('/root/autodl-tmp/zhjproject/DPSH/data/cifar/train.txt', train_index, fmt='%d')
    np.savetxt('/root/autodl-tmp/zhjproject/DPSH/data/cifar/test.txt', test_index, fmt='%d')
    np.savetxt('/root/autodl-tmp/zhjproject/DPSH/data/cifar/database.txt', database_index, fmt='%d')


"""
CIFAR数据集加载函数：cifar_dataset
    根据配置加载CIFAR-10数据集，进行数据划分（训练集、测试集、数据库集）和预处理，并返回数据加载器。
"""
def cifar_dataset(config):
    batch_size = config["batch_size"]

    # 2. 定义数据预处理步骤
    # 使用 transforms.Compose 组合了一系列的图像预处理步骤，包括调整图像大小、将图像转换为张量、归一化图像等。
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/root/autodl-tmp/zhjproject/DPSH/dataset/cifar/'
    
    # 3.加载数据集
    # 使用自定义的 MyCIFAR10 类加载训练集、测试集和数据库集数据。
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    # 4. 合并数据
    # 将训练集和测试集的数据合并，得到完整的数据 X 和标签 L。
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    # new step：已经生成了index了，保证不同的训练能够读取同一个index.txt
    with open("/root/autodl-tmp/zhjproject/DPSH/data/cifar/database.txt", "r") as f:
        database_index = [int(item) for item in f.readlines()]
    with open("/root/autodl-tmp/zhjproject/DPSH/data/cifar/train.txt", "r") as f:
        train_index = [int(item) for item in f.readlines()]
    with open("/root/autodl-tmp/zhjproject/DPSH/data/cifar/test.txt", "r") as f:
        test_index = [int(item) for item in f.readlines()]

    database_index = np.array(database_index)
    train_index = np.array(train_index)
    test_index = np.array(test_index)
    

    # 7. 更新数据集
    # 将划分后的数据和标签赋给相应的数据集对象。
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    # 8. 创建数据加载器
    # 使用 torch.utils.data.DataLoader 为训练集、测试集和数据库集创建数据加载器。
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    # 9. 返回结果
    # 返回训练集、测试集和数据库集的数据加载器，以及它们的大小。
    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


# ------------------------------------------------------------------------------------------------------------------------

# get_data(获取查询集 训练集 数据集库) 的必要函数
"""
图片列表类：ImageList
    该类用于加载图像列表，并应用预处理转换。
    用于 get_data 函数
"""
class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


"""
图片转换函数：image_transform
    该函数根据数据集类型（训练集或测试集）生成图像预处理步骤，包括随机水平翻转、裁剪、归一化等。
    用于 get_data 函数
"""
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


# ------------------------------------------------------------------------------------------------------------------------

"""
获取数据函数：get_data
    cifar_dataset 专门处理 CIFAR-10 数据集的加载和划分，
    而 get_data 函数则根据配置决定调用 cifar_dataset 还是使用 ImageList 和 DataLoader 处理其他数据集。
    根据配置加载指定的数据集，返回数据加载器和数据集大小。
"""
def get_data(config):
    # 1. 检查数据集类型
    # 如果数据集是 CIFAR-10，则调用 cifar_dataset 函数来加载数据。
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    # 2. 初始化字典
    # 初始化用于存储数据集对象和数据加载器的字典 dsets 和 dset_loaders。
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    # 3. 加载数据集
    # 遍历 train_set、test 和 database 三个数据集类型：
    #     使用 ImageList 类创建数据集对象。
    #     使用 image_transform 函数生成图像预处理步骤。
    #     使用 torch.utils.data.DataLoader 创建数据加载器。
    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=4)

    # 4. 返回结果
    # 返回训练集、测试集和数据库集的数据加载器，以及它们的大小。
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# ------------------------------------------------------------------------------------------------------------------------

# 评价函数、验证函数
"""
计算结果函数：compute_result
    计算数据加载器中所有数据的二进制哈希码和标签。
"""
def compute_result(dataloader, net, device):
    # 1. 初始化列表
    # bs 用于存储计算得到的二进制哈希码。
    # clses 用于存储对应的标签。
    bs, clses = [], []

    # 2. 设置网络为评估模式
    # net.eval() 将模型设置为评估模式，禁用 dropout 和 batch normalization 的训练行为。
    # 遍历数据加载器中的数据
    net.eval()

    # 3. 使用 tqdm 进度条库显示进度。
    # 对于每个批次的数据（图像和标签），将图像数据移动到指定设备（device），并通过网络前向传播计算哈希码。
    # 将输出移动到 CPU： data.cpu()
    # data 是模型在 GPU 上计算的结果，通过 data.cpu() 将其移动到 CPU 上。
    # 移动到 CPU 上后，可以进一步处理或保存。
    # 将计算得到的哈希码和标签分别添加到 bs 和 clses 列表中。
    # 合并结果
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())

    # 4. 使用 torch.cat 将所有批次的结果拼接成一个张量。
    # 对哈希码进行符号函数处理，确保它们是二进制（-1 或 1）。
    # 返回拼接后的哈希码和标签
    return torch.cat(bs).sign(), torch.cat(clses)

"""
计算汉明距离函数：CalcHammingDist
    计算两个二进制矩阵之间的汉明距离。
"""
def CalcHammingDist(B1, B2):
    # 1. 获取矩阵列数
    # q = B2.shape[1] 获取矩阵 B2 的列数。
    q = B2.shape[1]

    # 2. 计算汉明距离
    # np.dot(B1, B2.transpose()) 计算矩阵 B1 和 B2 的点积。
    # 0.5 * (q - np.dot(B1, B2.transpose())) 计算汉明距离。
    # 汉明距离是点积的补码，范围为 [0, q]。
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

"""
计算精确度函数：calculate_top_map
    计算平均准确率，适用于多标签图像检索任务。
    rB (retrieval Binary codes): 检索集哈希码矩阵
        检索集中所有图像的二进制哈希码矩阵，每一行表示一个图像的哈希码。
    qB (query Binary codes): 查询集哈希码矩阵
        查询集中所有图像的二进制哈希码矩阵，每一行表示一个图像的哈希码。
    retrievalL (retrieval Labels):
        检索集中所有图像的标签矩阵，每一行表示一个图像的标签。标签可以是多标签的，通常是二进制向量。
    queryL (query Labels):
        查询集中所有图像的标签矩阵，每一行表示一个图像的标签。标签可以是多标签的，通常是二进制向量。
    topk:
        用于计算平均准确率时要考虑的前 K 个检索结果。即在计算 MAP 时，只考虑每个查询的前 K 个最近邻。
"""
def CalcTopMap(rB, qB, retrievalL, queryL, topk):

    # 1. 初始化变量
    # num_query = queryL.shape[0] 获取查询集的数量。
    # topkmap = 0 初始化平均准确率为 0。    
    num_query = queryL.shape[0]
    topkmap = 0

    # 2. 遍历每个查询
    # 使用 tqdm 显示进度。
    # 计算当前查询与检索集的标签点积，得到 ground truth。
    # 计算当前查询与检索集的汉明距离。
    # 根据汉明距离对检索结果进行排序。
    # 取排序后的前 topk 个结果。
    # 计算 topk 的准确率并累加。
    for iter in tqdm(range(num_query)): # 遍历每个查询样本
        # 计算 ground truth 矩阵：
            # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            # 这一步计算查询样本与检索集所有样本的标签点积，得到一个布尔向量，表示每个检索样本是否与查询样本相关。
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # 计算汉明距离：
        #     hamm = CalcHammingDist(qB[iter, :], rB)
        #     计算查询样本与检索集所有样本的汉明距离。
        hamm = CalcHammingDist(qB[iter, :], rB)
        # 排序：
            # ind = np.argsort(hamm)
            # 按照汉明距离对检索结果进行排序，得到排序后的索引。
        ind = np.argsort(hamm)
        # 重排 ground truth：
            # gnd = gnd[ind]
            # 依据汉明距离的排序结果重排 ground truth。
        gnd = gnd[ind]
        # 取前 topk 个检索结果：
            # tgnd = gnd[0:topk]
            # 取排序后的前 topk 个检索结果
        tgnd = gnd[0:topk]
        # 计算 topk 内 relevant 的数量
        tsum = np.sum(tgnd).astype(int)
        # 如果没有 relevant 项，跳过该查询
        if tsum == 0:
            continue
        # 创建一个从 1 到 tsum 的数组
        count = np.linspace(1, tsum, tsum)

        # 找到 topk 内所有 relevant 项的索引
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        # 计算该查询的 topk 平均准确率
        topkmap_ = np.mean(count / (tindex))
        # 累加 topk 平均准确率
        topkmap = topkmap + topkmap_

    # 3. 计算平均准确率
    # topkmap = topkmap / num_query 计算平均准确率。
    topkmap = topkmap / num_query
    return topkmap

"""
该函数用于计算查询集和检索集之间的 Top-k 平均精度 (MAP) 和 PR 曲线（Precision-Recall Curve），并且考虑了内存的使用。
    qB (query Binary codes): 查询集的二进制哈希码矩阵。
    queryL (query Labels): 查询集的标签矩阵。
    rB (retrieval Binary codes): 检索集的二进制哈希码矩阵。
    retrievalL (retrieval Labels): 检索集的标签矩阵。
    topk: 计算平均准确率时要考虑的前 K 个检索结果。
"""
# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):

    # 1. 初始化变量
    # num_query: 查询集样本数量。
    # num_gallery: 检索集样本数量。
    # topkmap: 初始化的 Top-k 平均准确率。
    # prec 和 recall: 用于存储每个查询样本的精度和召回率。
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))


    for iter in tqdm(range(num_query)): # 遍历每个查询样本
        # 2.
        # gnd: 计算当前查询样本与检索集所有样本的标签点积，得到 ground truth。
        # hamm: 计算当前查询样本与检索集所有样本的汉明距离。
        # ind: 按照汉明距离对检索结果进行排序，得到排序后的索引。
        # gnd: 根据排序结果重排 ground truth。        
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        # 3. 计算 Top-k 精度和召回率
        # tgnd: 取排序后的前 topk 个检索结果。
        # tsum: 计算 topk 内 relevant 的数量。
        # count: 创建一个从 1 到 tsum 的数组。
        # all_sim_num: 计算 ground truth 中 relevant 项的总数。
        # prec_sum: 累积的精度和召回率计算。
        # prec 和 recall: 记录每个查询样本的精度和召回率。
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        # 4. 计算 Top-k 平均准确率
        # 确认最后的召回率为 1。
        # 计算当前查询样本的 topk 平均准确率，并累加到 topkmap。
        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    
    # 5. 计算全局 Top-k 平均准确率和 PR 曲线
    # 计算查询集的平均 topk 平均准确率。
    # 筛选有效的精度和召回率结果。
    # 计算累计的精度和召回率。
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
"""
该函数用于在验证集上评估模型的性能，计算 mAP，并在性能提升时保存最佳模型和结果。
    config: 配置字典，包含设备信息和其他配置参数。
    Best_mAP: 当前最佳的 mAP 值。
    test_loader: 测试数据集的数据加载器。
    dataset_loader: 数据库集的数据加载器。
    net: 神经网络模型。
    bit: 哈希码的位数。
    epoch: 当前训练的 epoch 数。
    num_dataset: 数据库集的样本数量。
"""
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]

    # 使用 compute_result 函数计算测试集和数据库集的二进制哈希码和标签。
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)
    # print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # 控制是否计算并保存 PR 曲线数据：
    # 根据配置中的 pr_curve_path，选择合适的函数计算 mAP 和 PR 曲线。
    # 如果配置中没有 pr_curve_path，则使用 CalcTopMap 计算 mAP，否则使用 CalcTopMapWithPR 计算 mAP 和 PR 曲线

    # 如果配置中不包含 pr_curve_path，则仅计算平均准确率（mAP）。
    if "pr_curve_path" not in  config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    # 如果配置中包含 pr_curve_path，则会计算 PR 曲线，并将结果保存到指定路径。
    # 保存 PR 曲线数据：当计算 PR 曲线时，将结果保存为 JSON 文件，便于后续可视化和分析。
    # 要实现可视化使用precision_recall_curve.py
    else:
        # need more memory
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])

    # 如果当前 mAP 超过了历史最佳值，则更新 Best_mAP 并保存最佳模型和对应的哈希码、标签数据到指定路径
    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            # 保存查询集和数据集库的二进制哈希码、标签码
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            # 保存模型参数
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}\n\n")
    print(config)
    return mAP, Best_mAP
