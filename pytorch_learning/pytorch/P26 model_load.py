import torchvision
import torch

#加载模型方式1 --对应保存模型方式11
model1=torch.load("vgg16_method1.pth",weights_only=False)
print(model1)
#加载模型方式1 --对应保存模型方式11
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth",weights_only=False))
print(vgg16)