import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 优先使用 Apple Silicon 的 MPS，其次回退到 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("当前训练设备：{}".format(device))

#第一步：准备数据集
train_data=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length 长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

#第二步：加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#第三步:搭建神经网络(已经单独放置于“model.py”python文件中,该文件要与本文件放置在同一个文件夹下)
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(in_features=1024, out_features=64),
#             nn.Linear(in_features=64, out_features=10)
#
#         )
#     def forward(self,x):
#         x=self.model(x)
#         return x

#第四步：创建网络模型
tudui=Tudui()
tudui = tudui.to(device)

#第五步：创建损失函数
loss_cross=nn.CrossEntropyLoss()

#第六步：创建优化器
learning_rate=0.01
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#第七步：设置训练网络的参数
#total_train_step记录训练次数
total_train_step=0
#total_test_step记录测试次数
total_test_step=0
#训练的轮数
epochs=10

#第九步：tensorboard
writer=SummaryWriter("P27")
#第八步循环 训练网络模型
for i in range(epochs):
    print("------第{}轮训练开始------".format(i+1))

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs=tudui(imgs)
        loss=loss_cross(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        if total_train_step%100==0:
            print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    #测试步骤开始
    tudui.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs=tudui(imgs)
            loss=loss_cross(outputs,targets)
            total_test_loss+=loss.item()

            #正确率
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    test_accuracy = total_accuracy.item() / test_data_size
    print("整体测试集上的正确率：{}".format(test_accuracy))


    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step+=1

    # 第十步：保存模型
    torch.save(tudui,"tudui_{}.pth".format(i+1))
    print("模型已保存")

writer.close()