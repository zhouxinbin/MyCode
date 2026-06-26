import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P9_tensorboard2 import writer

#准备的测试数据集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)


#测试数据集中第一张样本图片及target
img,target=test_data[0]

writer=SummaryWriter("p15")

for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step=step+1

writer.close()



