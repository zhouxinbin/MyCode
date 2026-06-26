import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(dataset,batch_size=64)
# input=torch.tensor([[1,2,0,3,1],
#                    [0,1,2,3,1],
#                    [1,2,1,0,0],
#                    [5,2,3,1,1],
#                    [2,1,0,1,1]],dtype=torch.float32)
# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.maxpool1 = nn.MaxPool2d(3,ceil_mode=True)


    def forward(self,input):
        output=self.maxpool1(input)
        return output


tudui=Tudui()
# output=tudui(input)
# print(output)
writer=SummaryWriter("p18")
step=0
for data in data_loader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=tudui(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()
