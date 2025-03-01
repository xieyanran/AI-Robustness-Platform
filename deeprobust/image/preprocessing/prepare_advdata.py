"""
This implementation is used to create adversarial dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from deeprobust.image.attack.pgd import PGD
import deeprobust.image.netmodels.resnet as resnet
from deeprobust.image.config import attack_params

def accuracy(output, target):
    """计算准确率"""
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct

def main():
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = resnet.ResNet18().to(device)
    model_path = "./train  "
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return
    model.eval()

    # 数据变换和加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('deeprobust/image/data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('deeprobust/image/data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=True)

    # 初始化变量
    train_acc = 0.0
    adv_acc = 0.0
    train_n = 0
    normal_data = []
    adv_data = []

    # PGD 攻击
    adversary = PGD(model, device=device)

    # 对训练集生成对抗样本
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        train_acc += accuracy(y_pred, y)

        # 生成对抗样本
        x_adv = adversary.generate(x, y, **attack_params['PGD_CIFAR10']).float()
        y_adv = model(x_adv)
        adv_acc += accuracy(y_adv, y)
        train_n += y.size(0)

        # 存储数据（转为 CPU 以节省 GPU 内存）
        normal_data.append(x.cpu())
        adv_data.append(x_adv.cpu())

    # 拼接数据
    normal_data = torch.cat(normal_data)
    adv_data = torch.cat(adv_data)

    # 打印结果
    print(f"Accuracy (normal): {train_acc / train_n * 100:.6f}%")
    print(f"Accuracy (PGD): {adv_acc / train_n * 100:.6f}%")

    # 保存数据和模型
    torch.save({"normal": normal_data, "adv": adv_data}, "data.tar")
    torch.save({"state_dict": model.state_dict()}, "cnn.tar")
    print("Saved data to 'data.tar' and model to 'cnn.tar'")

if __name__ == "__main__":
    main()