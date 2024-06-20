# DPSH PyTorch Implementation

DPSH论文的复现：包含 PyTorch 中 DPSH（深度成对监督哈希）的实现。该项目包括使用两种不同的网络架构（CNN-F 和 ResNet）在 CIFAR 和 NUS-WIDE 数据集上训练和评估深度哈希模型的代码。

## 参考代码
- [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch)
- [DPCHash_Baselines](https://github.com/Huenao/DPCHash_Baselines)

## 运行环境
- GPU运行
- Conda虚拟环境
- GPU版本的Pytorch

## 代码文件解释
1. **network.py**:我们定义了加载了 CNNF 的预训练网络，定义 CNNF 网络的结构，并且定义了前向传播函数。同时，我们还定义了用于优化检索效果的网络ResNet。
2. **2.	tools.py**: 我们定义了处理参数、加载数据集、处理数据集、计算结果、评估性能、验证与保存模型的功能，以供 DPSH.py 训练模型使用
3. **DPSH.py(4 个)**: 4 个 Python 文件，对应 CIFAR 数据集和 NUSWIDE 数据集分别在 2 个不同网络（CNN-F,ResNet）上的训练。只需直接运行即可开始训练模型（如有需要，请自行调整路径）。
4. **DPSH_test_generate.py**: 用于进行测试与生成 CIFAR 的训练集、测试集、数据库集。如果能够成功运行此代码，代表能够直接运行上述 4 个 DPSH.py 文件进行训练，并且成功生成了用于 CIFAR 的数据集。
5. **precision_recall_curve.py**: 根据生成的 log（位于 log 文件夹中），我们可以生成 Precision-Recall (PR) 曲线。
6. **loss_mAP_curve.ipynb**： 根据生成的LOSS.txt和mAP.txt（位于RecordTrain文件夹中），生成训练过程的损失曲线和验证的mAP曲线。
7. **demo_NUSWIDE.ipynb**: 指定好路径，可以展示某个基于 NUSWIDE21 数据集训练的模型的部分检索结果，作为演示示例。
8.	**demo_CIFAR.ipynb**: 指定好路径，可以展示某个基于 NUSWIDE21 数据集训练的模型的部分检索结果，作为演示示例。（因为 CIFAR 与 NUSWIDE21 读取数据集的方法不同，所以有两个 demo）


## 运行准备
1.	在DPSH下创建三个文件夹, 分别为log、RecordTrain、Save, 分别在这三个文件夹中创建两个文件夹，分别命名为ResNet和CNNF。这将分别对应用来保存log日志文件，loss和mAP记录文件，模型及其二进制哈希码表示。
2.	在DPSH下创建一个文件夹，命名为CNNFmodel，根据链接（http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat）下载预训练模型，将该预训练模型放入其中。
3.	在DPSH下创建文件夹datasets，在该文件夹中继续创建两个子文件夹，分别命名为cifar和nus-wide。下载NUSWIDE数据集，解压放入nus-wide文件夹中。
4.	您需要在DPSH.py以及tools.py中修改符合您运行环境的路径。


## 运行

生成用于CIFAR训练的train.txt以及test.txt以及database.txt并测试是否可以成功进行训练, run:
```sh
python DPSH_test_generate.py

如果成功，run：
```sh
python DPSH_CNNF_CIFAR.py
python DPSH_CNNF_NUSWIDE21.py
python DPSH_ResNet_CIFAR.py
python DPSH_ResNet_NUSWIDE21.py

## demo
您可以在两个demo.ipynb中指定路径并运行，来生成图像检索的示例查看模型的效果，如demo里的两个png。


