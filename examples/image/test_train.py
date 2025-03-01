import deeprobust.image.netmodels.train_model as trainmodel
trainmodel.train('vgg11','CIFAR10','cuda', 20)

# train models will be store in specific path
# model(option:'CNN', 'ResNet18', 'ResNet34', 'ResNet50', 'densenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
# data(option:'MNIST','CIFAR10')
# device(option:'cpu', 'cuda')
# training epoch
