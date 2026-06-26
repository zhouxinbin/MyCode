import torchvision
from sympy.printing.pytorch import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,drop_last=False)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1=nn.Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024,out_features=64),
            Linear(in_features=64,out_features=10)
        )

    def forward(self,x):

        x=self.model1(x)
        return x

loss_cross=nn.CrossEntropyLoss()

tudui=Tudui()
#定义一个优化器
optim=torch.optim.SGD(tudui.parameters(),lr=0.01)

for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=tudui(imgs)
        result_loss=loss_cross(outputs,targets)

        #网络模型中 参数对应梯度调为0
        optim.zero_grad()
        #利用反向传播获取每个参数的梯度
        result_loss.backward()
        #调用优化器 进行调优
        optim.step()
        running_loss += result_loss
    print(running_loss)