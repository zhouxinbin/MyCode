from sympy.printing.pytorch import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1=Conv2d(3,32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1=nn.MaxPool2d(kernel_size=2)
        # self.conv2=Conv2d(32,32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2=nn.MaxPool2d(kernel_size=2)
        # self.conv3=Conv2d(32,64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3=nn.MaxPool2d(kernel_size=2)
        # self.flatten=Flatten()
        # self.linear1=Linear(in_features=1024,out_features=64)
        # self.linear2=Linear(in_features=64,out_features=10)

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x=self.model1(x)
        return x


tudui=Tudui()
print(tudui)
input=torch.ones(64,3,32,32)
output=tudui(input)
print(output.shape)

writer=SummaryWriter("p22")
writer.add_graph(tudui,input)
writer.close()




