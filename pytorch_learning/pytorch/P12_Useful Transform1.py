from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")

img=Image.open("data/train/ants_image/0013035.jpg")

# Totensor使用
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("Totensor",img_tensor)

#Normalize归一化
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)

print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)
writer.close()
