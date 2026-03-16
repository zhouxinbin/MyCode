
from torchvision import transforms
from PIL import Image

#tensor数据类型
#通过transform.ToTensor解决两个问题
# 1.transform如何使用
# 2.Tensor数据类型和其他类型有什么区别，及为什么需要该类型

#绝对路径："D:\Study\Mystudy\pytorch_learning\data\train\ants_image\0013035.jpg"
#相对路径："data/train/ants_image/0013035.jpg"


img_path="data/train/ants_image/0013035.jpg"
img=Image.open(img_path)


tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

print(tensor_img)