from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer=SummaryWriter("logs")

img_path="data/train/ants_image/0013035.jpg"
img=Image.open(img_path)

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)


writer.add_image("Tensor_img",tensor_img,0)
writer.close()
  