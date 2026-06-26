import torch
import torchvision
from torch import nn
from torch.ao.nn.quantized.functional import conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
DataLoader=DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

tudui=Tudui()
writer=SummaryWriter("p17")

step=0
for data in DataLoader:
    imgs,targets=data
    output=tudui(imgs)
    print(output.shape)
    writer.add_images("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("outout",output,step)
    step=step+1
writer.close()