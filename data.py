from torchvision.datasets import MNIST              #导入MNIST
import torchvision.transforms as transforms         #导入transforms包改变图像属性
from torch.utils.data import DataLoader             #导入DataLoader类
import numpy 

data_train = MNIST('D:/数据结构学习/LeNet_master/data/train', transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ]), download=True)              #从internet上下载MNIST训练数据集，并Resize成32x32大小，转为Tensor

data_test = MNIST('D:/数据结构学习/LeNet_master/data/test', train=False, transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ]), download=True)              #从internet上下载MNIST测试数据集，并Resize成32x32大小，转为Tensor    

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0) #8个线程处理，随机抽取，batch=256
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)



## test_data_load       测试图片时使用
import matplotlib.pyplot as plt
figure = plt.figure()
num_pic = 60
rows = 5
cols = num_pic/5 + 1

for imgs, tragets in data_train_loader:
    break

for index in range(num_pic):
    plt.subplot(rows, cols, index+1)
    plt.axis('off')
    img = imgs[index, ...]
    print(img.shape)
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.show()