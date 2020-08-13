import torch 
import torch.nn as nn

from model import LeNet
from torchvision.datasets import MNIST              #导入MNIST
import torchvision.transforms as transforms         #导入transforms包改变图像属性
from torch.utils.data import DataLoader             #导入DataLoader类
import numpy 
import random
import heapq
import matplotlib.pyplot as plt

import argparse

# data_train = MNIST('D:/数据结构学习/LeNet_master/data/train', transform=transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.ToTensor()
#     ]), download=True)              #从internet上下载MNIST训练数据集，并Resize成32x32大小，转为Tensor

parser = argparse.ArgumentParser(description='PyTorch LeNet Inference')     #使用argparse库实现 超参数的命令行 输入

parser.add_argument('--batch_size', '-b', default=1024, type=int, help='BatchSize')
args = parser.parse_args()                                                             #若想调用parser的中的参数，直接调用args.参数

data_test = MNIST('D:/数据结构学习/LeNet_master/data/test', train=False, transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ]), download=True)              #从internet上下载MNIST测试数据集，并Resize成32x32大小，转为Tensor    


# data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8) #8个线程处理，随机抽取，batch=256
data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

### LeNet工程推理代码部分

save_info_path = 'D:/数据结构学习/LeNet_master/model_save/model.pth'
save_info = torch.load(save_info_path)
model = LeNet()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(save_info["model"])
model.eval()

test_loss = 0
correct = 0
total = 0
epoch_num =5
outputs_plt = []
inputs_plt = []
target_plt = []

with torch.no_grad():       ## 测试时不需要开启计算图
    for epoch in range(epoch_num):
        for batch_idx, (inputs, targets) in enumerate(data_test_loader):
            # print(inputs.shape)
            # print(targets.shape)
            inputs_plt.append(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs_plt.append(outputs)
            target_plt.append(targets)

            test_loss += loss.item()
            _, predict = outputs.max(1)
            # print(targets)
            # print(predict)
            total += targets.size(0)
            correct += predict.eq(targets).sum().item()
            print(batch_idx, len(data_test_loader), 'Loss: %.3f | Acc: %.3f%%(%d,%d)'  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    num = random.randint(0, (9)*epoch_num)
    num1 = random.randint(0, args.batch_size) 
    print(len(inputs_plt))
    print(len(outputs_plt))
    inputs_show = inputs_plt[num][num1,:,:]         ##找到第num个inputs，并取num1通道的图片(num1为随机值)
    print(inputs_show.shape)
    value, predict_show = outputs_plt[num].max(1)
    predict_show = predict_show.numpy()
    value = value.numpy()
    print(predict_show[num1])       #预测的数       
    print(value[num1])              #预测概率
    print(target_plt[num][num1])   #目标值
    # m= max(value)
    # print(m)
    # index = numpy.where(value==m)
    # print(index)
    # print(predict_show[index])
    
    # plt.figure(1)
    # value_former_5 = heapq.nlargest(10, value)
    # index = []
    # for i in value_former_5:
    #     index.append(numpy.where(value==i))   ##寻找出value的前10大值，同时求出索引
    
    # predict_former_5 = []
    
    # for i in index:
    #     predict_former_5.append(predict_show[i])
        
    # print(value_former_5)
    # print(predict_former_5)

    # fig,ax=plt.subplots()
    # ax.bar([i+1 for i in range(len(value_former_5))], value_former_5, width=0.5)
    # ax.set_xlabel("all numbers")
    # ax.set_ylabel("confidence")
    # ax.set_title("pridect")

    # plt.xticks([i+1 for i in range(len(predict_former_5))], predict_former_5, rotation=90)


    # plt.figure(1)
    if predict_show[num1] == target_plt[num][num1]:
        correct_or_not = 'True'
    else:
        correct_or_not = 'False'
    plt.imshow(inputs_show.numpy().squeeze(), cmap='gray_r')
    plt.xlabel('The predict Number is {}, The judgement is '.format(predict_show[num1]) + ' ' +correct_or_not)
    # squeeze()[1022:-1,:,:]
    plt.show()
    



