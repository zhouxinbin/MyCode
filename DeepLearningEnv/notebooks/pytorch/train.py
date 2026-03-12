import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *

# 下载数据集
train_data = torchvision.datasets.CIFAR10("/workspace/data/cifar-10-batches-py", 
                                          train=True, 
                                          transform=torchvision.transforms.ToTensor(), 
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/workspace/data/cifar-10-batches-py", 
                                          train=False, 
                                          transform=torchvision.transforms.ToTensor(), 
                                          download=True)

print("训练数据集的长度: {}".format(len(train_data)))
print("测试数据集的长度: {}".format(len(test_data)))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
cifar = CIFAR()
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(cifar.parameters(), lr=0.01)

writer = SummaryWriter("/workspace/notebooks/pytorch/cifar_logs")
# tensorboard --logdir=/workspace/notebooks/pytorch/cifar_logs

total_train_step = 0
total_test_step = 0
epoch = 20


for i in range(epoch):
    cifar.train()
    # 训练
    print("------第 {} 轮训练开始------".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        output = cifar(imgs)
        loss = loss_function(output, targets)
        # 学习
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("第 {} 轮 {} 次训练 Loss = {}".format(i + 1, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试评估
    cifar.eval()
    loss_sum = 0.
    total_acc = 0
    with torch.no_grad(): # 停止调优
        for data in test_dataloader:
            imgs, targets = data
            output = cifar(imgs)
            loss = loss_function(output, targets)
            loss_sum = loss_sum + loss.item()

            acc = (output.argmax(1) == targets).sum()
            total_acc = total_acc + acc


    print("第 {} 轮测试数据集损失之和 = {}".format(i + 1, loss_sum))
    writer.add_scalar("test_loss", loss_sum, total_test_step)
    print("整体正确率 {}".format(total_acc/len(test_data)))
    writer.add_scalar("acc_rate", total_acc/len(test_data), total_test_step)
    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(cifar, "/workspace/notebooks/pytorch/models/cifar{}.pth".format(i))

writer.close()