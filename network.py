# from google.colab import drive
# drive.mount('/content/drive')

# import sys

# sys.path.append('/content/drive/MyDrive/DPSH')

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import scipy.io as sio
import numpy as np



# 加载 .mat 文件
CNN_F = "/root/autodl-tmp/zhjproject/DPSH/CNNFmodel/imagenet-vgg-f.mat"
cnnf = sio.loadmat(CNN_F)
weights = cnnf["layers"][0]

# 定义辅助函数来创建网络层
def make_lrn(i):
    param = weights[i][0][0][2][0]
    lrn = nn.LocalResponseNorm(int(param[0]), param[1], param[2], param[3])
    return lrn

def make_pool(i):
    k_size = weights[i][0][0][3][0]
    stride = weights[i][0][0][4][0]
    pad = weights[i][0][0][5][0]
    padding = nn.ZeroPad2d(tuple(pad.astype(np.int32)))
    pool_type = weights[i][0][0][2][0]

    if pool_type == 'max':
        pool = nn.MaxPool2d(tuple(k_size), stride=tuple(stride))
    else:
        pool = nn.AvgPool2d(tuple(k_size), stride=tuple(stride))

    return nn.Sequential(padding, pool)

def make_conv(i):
    k_weight = weights[i][0][0][2][0][0]
    bias = weights[i][0][0][2][0][1]
    k_shape = weights[i][0][0][3][0]
    inchannel = k_shape[-2]
    outchannel = k_shape[-1]
    pad = weights[i][0][0][4][0]
    stride = weights[i][0][0][5][0][0]

    conv = nn.Conv2d(inchannel, outchannel, k_shape[0], stride, pad[0])
    conv.weight.data = torch.from_numpy(k_weight.transpose((3, 2, 0, 1)))
    conv.bias.data = torch.from_numpy(bias.reshape(-1))

    return conv

# 定义模型结构
class CNNF(nn.Module):
    def __init__(self, hash_bit):
        super(CNNF, self).__init__()
        self.conv1 = make_conv(0)
        self.relu1 = nn.ReLU(inplace=True)
        self.lrn1 = make_lrn(2)
        self.pool1 = make_pool(3)
        self.conv2 = make_conv(4)
        self.relu2 = nn.ReLU(inplace=True)
        self.lrn2 = make_lrn(6)
        self.pool2 = make_pool(7)
        self.conv3 = make_conv(8)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = make_conv(10)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = make_conv(12)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = make_pool(14)
        # full6: 4096
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        # full7: 4096
        self.fc7 = nn.Linear(4096, 4096)
        # hash layer
        self.hash_layer = nn.Linear(4096, hash_bit)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.hash_layer(x)
        return x



"""
AlexNet 类:

初始化函数 __init__:
    构建了一个基于AlexNet的神经网络，并在其顶部添加了一个哈希层。
    model_alexnet 使用 torchvision 库中的 alexnet 函数加载预训练的AlexNet模型。
    提取了AlexNet的特征提取部分 (features)。
    定义了两个全连接层 cl1 和 cl2，其权重和偏置分别从 model_alexnet 的分类器层 (classifier) 的对应层复制。
    定义了哈希层 (hash_layer)，它由一个Dropout层、两个全连接层和一个ReLU激活函数，以及最终的一个线性层组成，其输出维度是 hash_bit。
前向传播函数 forward:
    输入 x 通过 features 提取特征。
    将特征展平为一维向量。
    通过 hash_layer 生成哈希码。
"""
class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x



"""
ResNet 类:
初始化函数 __init__:
    构建了一个基于ResNet的神经网络，并在其顶部添加了一个哈希层。
    model_resnet 使用 torchvision 库中的 resnet 函数加载预训练的ResNet模型。
    提取了ResNet的特征提取部分 (feature_layers)。
    定义了一个线性层 hash_layer，其输出维度是 hash_bit。对其权重和偏置进行了初始化。
前向传播函数 forward:
    输入 x 通过 feature_layers 提取特征。
    将特征展平为一维向量。
    通过 hash_layer 生成哈希码。
"""
resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
