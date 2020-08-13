import torch
import torch.nn as nn
from model import LeNet
import tqdm
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST              #导入MNIST
import torchvision.transforms as transforms         #导入transforms包改变图像属性
from torch.utils.data import DataLoader             #导入DataLoader类
import numpy 

data_train = MNIST('D:/数据结构学习/LeNet_master/data/train', transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ]), download=True)              #从internet上下载MNIST训练数据集，并Resize成32x32大小，转为Tensor

# data_test = MNIST('D:/数据结构学习/LeNet_master/data/test', train=False, transform=transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.ToTensor()
#     ]), download=True)              #从internet上下载MNIST测试数据集，并Resize成32x32大小，转为Tensor    

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0) #8个线程处理，随机抽取，batch=256
# data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
##  数据载入
# from data import data_train_loader

#### 定义模型训练参数部分

model = LeNet()
model.cpu()
model.train() ## 切换到模型的训练模式
lr = 0.001
loss_define = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=5e-4)
train_loss = 0  ## 训练损失  
correct = 0     ## 识别正确个数
total = 0       ## 总个数
# momentum=0.9,

epoch_num = 10
# int(input('input epoch:'))
loss_plot = []

for epoch in range(epoch_num):
    #### 
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        output = model(inputs)       ##识别10个手写数字（0~9），因此output输出10个概率值
        loss = loss_define(output, targets) ##前向传播计算出损失
        loss.backward()                    ##对损失进行反向传播
        optimizer.step()                   ##根据学习率等超参数进行梯度更新（本文使用Adam）

    ###  未完待续，8月4日
        train_loss += loss.item()        # 该步的训练损失
        _, predict = output.max(1)       # predict输出的将是output中的最大的一个概率
        total += targets.size(0)         # 参与训练的总样本数   
        correct += predict.eq(targets).sum().item()     #预测正确的数量(使output与target对比)
        print(batch_idx, len(data_train_loader), 'Loss: %.3f | Acc: %.3f%%(%d,%d)'  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    loss_plot.append(train_loss/(batch_idx+1))
        
plt.plot(range(epoch_num), loss_plot,'--')
plt.show()
## 模型保存
save_info = {
    "epoch_num": epoch_num,
    "optimizer": optimizer.state_dict(),
    "model": model.state_dict(),
}
# save_dict = model.state_dict()
torch.save(save_info, 'D:/数据结构学习/LeNet_master/model_save/model.pth')




