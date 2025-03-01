# one-time-shot
import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from deeprobust.image import utils
import random
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.deepfool import DeepFool
from deeprobust.image.attack.lbfgs import LBFGS
from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.onepixel import Onepixel
from deeprobust.image.attack.YOPOpgd import FASTPGD

app = Flask(__name__, template_folder='template')

# 全局模型变量
model = None


def load_net(attack_model, filename, path, device):
    if attack_model == "CNN":
        from deeprobust.image.netmodels.CNN import Net
        model = Net()
    elif attack_model == "ResNet18":
        from deeprobust.image.netmodels.resnet import ResNet18
        model = ResNet18()
    elif attack_model == "ResNet34":
        from deeprobust.image.netmodels.resnet import ResNet34
        model = ResNet34()
    elif attack_model == "ResNet50":
        from deeprobust.image.netmodels.resnet import ResNet50
        model = ResNet50()
    elif attack_model == "ResNet101":
        from deeprobust.image.netmodels.resnet import ResNet101
        model = ResNet101()
    elif attack_model == "ResNet152":
        from deeprobust.image.netmodels.resnet import ResNet152
        model = ResNet152()
    elif attack_model == "densenet":
        from deeprobust.image.netmodels.densenet import DenseNet
        model = DenseNet()
    elif attack_model == "vgg11":
        from deeprobust.image.netmodels.vgg import VGG
        model = VGG(vgg_name="VGG11")
    elif attack_model == "vgg13":
        from deeprobust.image.netmodels.vgg import VGG
        model = VGG(vgg_name="VGG13")
    elif attack_model == "vgg16":
        from deeprobust.image.netmodels.vgg import VGG
        model = VGG(vgg_name="VGG16")
    elif attack_model == "vgg19":
        from deeprobust.image.netmodels.vgg import VGG
        model = VGG(vgg_name="VGG19")
    else:
        raise ValueError(f"Unsupported model: {attack_model}")

    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model.to(device)


def generate_dataloader(dataset, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = 'data'

    if dataset == "MNIST":
        # 检查 MNIST 数据文件是否存在
        mnist_dir = os.path.join(data_dir, 'MNIST', 'raw')
        mnist_files = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        data_exists = all(os.path.exists(os.path.join(mnist_dir, f)) for f in mnist_files)

        if data_exists:
            print("MNIST dataset found locally, loading from disk.")
            download = False
        else:
            print("MNIST dataset not found, downloading...")
            download = True

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, download=download, transform=transform),
            batch_size=batch_size, shuffle=True)
        print("MNIST dataset loaded successfully.")

    elif dataset == "CIFAR10":
        # 检查 CIFAR10 数据文件是否存在
        cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        cifar_file = 'cifar-10-python.tar.gz'  # 下载后的压缩文件
        data_exists = os.path.exists(os.path.join(data_dir, cifar_file)) or os.path.exists(cifar_dir)

        if data_exists:
            print("CIFAR10 dataset found locally, loading from disk.")
            download = False
        else:
            print("CIFAR10 dataset not found, downloading...")
            download = True

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, download=download, transform=transform),
            batch_size=batch_size, shuffle=True)
        print("CIFAR10 dataset loaded successfully.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return test_loader


def run_attack(attack_method, batch_size, batch_num, device, test_loader, random_targeted=False, target_label=-1,
               **kwargs):
    global model
    test_loss = 0
    correct = 0
    count = 0
    class_num = 10

    for count, (data, target) in enumerate(test_loader):
        if count >= batch_num:
            break
        print(f"Processing batch: {count}")
        data, target = data.to(device), target.to(device)

        if random_targeted:
            r = list(range(0, target[0].item())) + list(range(target[0].item() + 1, class_num))
            target_label = random.choice(r)
            adv_example = attack_method.generate(data, target, target_label=target_label, **kwargs)
        elif target_label >= 0:
            adv_example = attack_method.generate(data, target, target_label=target_label, **kwargs)
        else:
            adv_example = attack_method.generate(data, target, **kwargs)

        output = model(adv_example)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    batch_num = count + 1
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / (batch_num * batch_size)
    return test_loss, accuracy


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/run_attack', methods=['POST'])
def execute_attack():
    global model
    try:
        # 从表单获取参数
        attack_method_name = request.form.get('attack_method', 'PGD')
        attack_model = request.form.get('attack_model', 'CNN')
        path = request.form.get('path', './trained_models/')
        filename = request.form.get('file_name', 'MNIST_CNN_epoch_20.pt')
        dataset = request.form.get('dataset', 'MNIST')
        batch_size = int(request.form.get('batch_size', 1000))
        batch_num = int(request.form.get('batch_num', 1000))
        epsilon = float(request.form.get('epsilon', 0.3))
        device = request.form.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        random_targeted = request.form.get('random_targeted', 'False') == 'True'
        target_label = int(request.form.get('target_label', -1))

        # 加载模型（仅首次加载）
        if model is None:
            model = load_net(attack_model, filename, path, device)

        # 加载数据集
        test_loader = generate_dataloader(dataset, batch_size)

        # 初始化攻击方法
        if attack_method_name == "PGD":
            attack_method = PGD(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "FGSM":
            attack_method = FGSM(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "C&&W":
            attack_method = CarliniWagner(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "DeepFool":
            attack_method = DeepFool(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "LBFGS":
            attack_method = LBFGS(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "NATTACK":
            attack_method = NATTACK(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "Onepixel":
            attack_method = Onepixel(model, device=device)
            kwargs = {'epsilon': epsilon}
        elif attack_method_name == "FASTPGD":
            attack_method = FASTPGD(model, device=device)
            kwargs = {'epsilon': epsilon}
        else:
            return jsonify({'error': f"Unsupported attack method: {attack_method_name}"}), 400

        # 运行攻击
        test_loss, accuracy = run_attack(
            attack_method, batch_size, batch_num, device, test_loader,
            random_targeted=random_targeted, target_label=target_label, **kwargs
        )

        # 返回结果
        result = {
            'test_loss': f'{test_loss:.4f}',
            'accuracy': f'{accuracy:.0f}',
            'parameters': {
                'attack_method': attack_method_name,
                'attack_model': attack_model,
                'path': path,
                'file_name': filename,
                'dataset': dataset,
                'batch_size': batch_size,
                'batch_num': batch_num,
                'epsilon': epsilon,
                'device': device,
                'random_targeted': random_targeted,
                'target_label': target_label
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)