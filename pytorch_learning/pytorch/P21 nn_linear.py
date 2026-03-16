import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(dataset,batch_size=64,drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1= Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

tudui=Tudui()


for data in data_loader:
    imgs,targets=data
    #flatten:展平为一行
    output=torch.flatten(imgs)
    output =tudui(output)
    print(output.shape)
