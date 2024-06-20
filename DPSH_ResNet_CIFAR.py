from utils.tools import *
from network import *
import torch
import torch.optim as optim
import time

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "alpha": 0.1,
        "need_PR": False,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DPSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        # "net": CNNF,
        # "net": AlexNet,
        "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "nuswide_10",
        # "dataset": "nuswide_21",
        "epoch": 30,
        "test_map": 3, # 每过 n 轮进行一次验证
        "save_path": "/root/autodl-tmp/zhjproject/DPSH/save/ResNet/CIFAR",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [12,24,32,48],
    }
    config = config_dataset(config)
    return config


class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = config["alpha"] * (u - u.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss
    


def train_val(config, bit):
    device = config["device"]

    """
    train_loader：训练数据加载器。
    test_loader：测试数据加载器。
    dataset_loader：完整数据集加载器。
    num_train：训练样本数量。
    num_test：测试样本数量。
    num_dataset：完整数据集样本数量。
    """
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DPSHLoss(config, bit)

    Best_mAP = 0

    # 记录trainloss 
    trainloss_list = []
    # 记录mAP
    mAP_list = []

    # begin to train
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # 计算并打印平均训练损失
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.5f" % (train_loss))
        trainloss_list.append(train_loss)

        # 定期验证模型（每过test_map轮验证一次，并且如果比之前的模型好，则保存模型）
        if (epoch + 1) % config["test_map"] == 0:
            epochmAP, Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            mAP_list.append(epochmAP)
    


    # 将 trainloss_list, mAP_list 输出
    loss_output_path = f"/root/autodl-tmp/zhjproject/DPSH/RecordTrain/ResNet/loss_DPSH_ResNet_{config['dataset']}_{bit}.txt"
    mAP_output_path = f"/root/autodl-tmp/zhjproject/DPSH/RecordTrain/ResNet/mAP_DPSH_ResNet_{config['dataset']}_{bit}.txt"

    # 写入loss
    with open(loss_output_path, 'w') as file:
    # 遍历列表中的每个元素
        for item in trainloss_list:
            # 将元素转换为字符串并写入文件，后面加上换行符
            file.write(f'{item}\n')

    # 写入mAP
    with open(mAP_output_path, 'w') as file:
    # 遍历列表中的每个元素
        for item in mAP_list:
            # 将元素转换为字符串并写入文件，后面加上换行符
            file.write(f'{item}\n')




def main():
    config = get_config()
    for bit in config["bit_list"]:
        config["save_path"] = f"/root/autodl-tmp/zhjproject/DPSH/save/ResNet/CIFAR_{bit}"
        config["pr_curve_path"] = f"/root/autodl-tmp/zhjproject/DPSH/log/ResNet/DPSH_ResNet_{config['dataset']}_{bit}.json"
        train_val(config, bit)

if __name__ == "__main__":
    main()