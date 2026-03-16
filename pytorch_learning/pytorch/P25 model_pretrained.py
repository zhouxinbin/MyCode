import torchvision
from scipy.datasets import download_all
from torch import nn
from torch.utils.data import DataLoader


vgg16_false=torchvision.models.vgg16(pretrained=False)
vgg16_true=torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#添加网络层数
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)
#修改网络层数
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)