import torchvision
from torch.utils.tensorboard import SummaryWriter

from P9_tensorboard2 import writer

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform,download=True)
# print(test_set[0])
# print(test_set.classes)
# img,targets=test_set[0]
# print(img,targets)
# print(test_set.classes[targets])
# img.show()

writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)
writer.close()
