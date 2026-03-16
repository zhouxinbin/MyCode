from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
#add_scalar:用来画出折线图 2*i；y轴 i：x轴
    writer.add_scalar("y=2x", 2*i,i)

writer.close()
