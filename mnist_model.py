import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import sys
sys.path.append("..")
from config import batch_size
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,5)
        self.conv3=nn.Conv2d(20,40,3)
        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(40,10)
    def forward(self,x):
        in_size=x.size(0)
        x=fun.relu(self.mp(self.conv1(x)))
        x=fun.relu(self.mp(self.conv2(x)))
        x=fun.relu(self.mp(self.conv3(x)))
        x=x.view(in_size,-1)
        x=self.fc(x)
        return fun.log_softmax(x)
model=Network()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


