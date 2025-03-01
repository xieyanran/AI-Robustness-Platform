from deeprobust.image.defense.pgdtraining import PGDtraining
from deeprobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
from deeprobust.image.netmodels.CNN import Net
from deeprobust.image.config import defense_params

# PGD 对抗训练

"""
LOAD DATASETS
"""

train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=256,
                shuffle=True)

test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('deeprobust/image/defense/data', train=False,
            transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=256,
            shuffle=True)


"""
TRAIN DEFENSE MODEL
"""

print('====== START TRAINING =====')

model = Net()

defense = PGDtraining(model, 'cuda')
defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])
# defense_params：从 deeprobust.image.config 导入的一个字典，包含预定义的防御方法参数

print('====== FINISH TRAINING =====')

