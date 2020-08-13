import torch 
import torch.nn as nn

class LeNet(nn.Module):
    '''
    LeNet的实现,实现特征提取部分
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3).cpu()
        self.pooling1 = nn.MaxPool2d(3, stride=2).cpu()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3).cpu()
        self.pooling2 = nn.MaxPool2d(2, stride=2).cpu()
        self.fc1 = nn.Linear(16*6*6, 120).cpu()
        self.fc2 = nn.Linear(120,84).cpu()
        self.fc3 = nn.Linear(84,10).cpu()

    def forward(self, x):
        out = self.conv1(x).cpu()
        out = torch.relu(out).cpu()
        out = self.pooling1(out).cpu()
        out = self.conv2(out).cpu()
        out = torch.relu(out).cpu()
        out = self.pooling2(out).cpu()
        out = out.view(out.size(0), -1).cpu()                      #测试张量大小
        out = self.fc1(out).cpu()
        out = torch.relu(out).cpu()
        out = self.fc2(out).cpu()
        out = torch.relu(out).cpu()
        out = self.fc3(out).cpu()
        return out

if __name__ == "__main__":
    
    model = LeNet()
    input_tensor = torch.randn(1,1,32,32)
    out = model(input_tensor)
    print(out.shape)