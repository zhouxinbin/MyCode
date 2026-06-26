import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)

#保存模型方式1（保存模型的结构+模型参数）
torch.save(vgg16,"vgg16_method1.pth")

#保存模型方式2(仅保存模型的参数)官方推荐
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
