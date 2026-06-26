from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("data/train/ants_image/0013035.jpg")
#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
#img PIL类型--resize--img_resize PIL类型
img_resize=trans_resize(img)

# img_resize PIL类型--totensor-- img——tensor tensor类型
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img_resize)
writer.add_image("Resize",img_tensor,0)


#compose
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize2",img_resize_2,0)

writer.close()