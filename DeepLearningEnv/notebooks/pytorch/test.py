import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

image_path = "notebooks/pytorch/imgs/image.png"
image = Image.open(image_path)
image = image.convert("RGB")
# print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)

# 模型信息
class CIFAR(nn.Module):
    def __init__(self):
        super(CIFAR, self).__init__()
        self.sequential = Sequential(
            Conv2d(3, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding = 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        return self.sequential(input)
    

# 导入模型
model = torch.load("/workspace/notebooks/pytorch/models/cifar19.pth", weights_only=False)
# print(model)
# 变换模型需要的尺寸
image = torch.reshape(image, (1, 3, 32, 32))
# 开始验证
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(classes[output.argmax(1).item()])