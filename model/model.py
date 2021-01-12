import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import sys
sys.path.append("..")
from config import batch_size
class Network2(nn.Module):
    def __init__(self,output_maps=9):
        super(Network2,self).__init__()
        self.bn03=nn.BatchNorm2d(3)        
        self.bn080=nn.BatchNorm2d(8)
        self.bn160=nn.BatchNorm2d(16)
        self.bn240=nn.BatchNorm2d(24)
        self.bn320=nn.BatchNorm2d(32)
        self.bn081=nn.BatchNorm2d(8)
        self.bn161=nn.BatchNorm2d(16)
        self.bn241=nn.BatchNorm2d(24)
        self.bn321=nn.BatchNorm2d(32)
        self.bn082=nn.BatchNorm2d(8)
        self.bn162=nn.BatchNorm2d(16)
        self.bn242=nn.BatchNorm2d(24)
        self.bn322=nn.BatchNorm2d(32)
        self.bn083=nn.BatchNorm2d(8)
        self.bn163=nn.BatchNorm2d(16)
        self.bn243=nn.BatchNorm2d(24)
        self.bn323=nn.BatchNorm2d(32)
        self.bn09=nn.BatchNorm2d(9)
        self.bn16=nn.BatchNorm2d(16)
        self.bn24=nn.BatchNorm2d(24)
        self.bn26=nn.BatchNorm2d(8+output_maps*2)   
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5,padding=2)
        self.conv2=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,padding=2)
        self.conv3=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=5,padding=2)
        self.conv4=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)   
        self.conv010=nn.Conv2d(in_channels=24,out_channels=8,kernel_size=5,padding=2)
        self.conv011=nn.Conv2d(in_channels=24,out_channels=8,kernel_size=5,padding=2)
        self.conv012=nn.Conv2d(in_channels=24,out_channels=8,kernel_size=5,padding=2)
        self.conv020=nn.Conv2d(in_channels=48,out_channels=16,kernel_size=5,padding=2)
        self.conv021=nn.Conv2d(in_channels=48,out_channels=16,kernel_size=5,padding=2)
        self.conv022=nn.Conv2d(in_channels=48,out_channels=16,kernel_size=5,padding=2)
        self.conv030=nn.Conv2d(in_channels=72,out_channels=24,kernel_size=5,padding=2)
        self.conv031=nn.Conv2d(in_channels=72,out_channels=24,kernel_size=5,padding=2)
        self.conv032=nn.Conv2d(in_channels=72,out_channels=24,kernel_size=5,padding=2)        
        self.conv040=nn.Conv2d(in_channels=56,out_channels=32,kernel_size=5,padding=2)
        self.conv041=nn.Conv2d(in_channels=56,out_channels=32,kernel_size=5,padding=2)
        self.conv042=nn.Conv2d(in_channels=56,out_channels=32,kernel_size=5,padding=2)
        self.conv13=nn.Conv2d(in_channels=56,out_channels=24,kernel_size=5,padding=2)
        self.conv12=nn.Conv2d(in_channels=40,out_channels=16,kernel_size=5,padding=2)
        self.conv11=nn.Conv2d(in_channels=24,out_channels=8+output_maps*2,kernel_size=5,padding=2)
        self.conv10=nn.Conv2d(in_channels=8+output_maps*2,out_channels=output_maps,kernel_size=5,padding=2)
        self.us=nn.UpsamplingNearest2d(scale_factor=2)
        self.mp=nn.AvgPool2d(2)        
        self.mp2=nn.MaxPool2d(2)     
    def forward(self,x): 
        x1=self.bn03(x)            
        x2=self.mp(x1)
        x3=self.mp(x2)
        x4=self.mp(x3)
        x1=fun.relu(self.bn080(self.conv1(x1)))
        x2=fun.relu(self.bn160(self.conv2(x2)))
        x3=fun.relu(self.bn240(self.conv3(x3)))
        x4=fun.relu(self.bn320(self.conv4(x4)))       
        
        y1=torch.cat([x1,self.us(x2)],1)
        y2=torch.cat([self.mp2(x1),x2,self.us(x3)],1)
        y3=torch.cat([self.mp2(x2),x3,self.us(x4)],1)
        y4=torch.cat([self.mp2(x3),x4],1)            
        x1=fun.relu(self.bn081(self.conv010(y1)))
        x2=fun.relu(self.bn161(self.conv020(y2)))
        x3=fun.relu(self.bn241(self.conv030(y3)))
        x4=fun.relu(self.bn321(self.conv040(y4)))
        y1=torch.cat([x1,self.us(x2)],1)
        y2=torch.cat([self.mp2(x1),x2,self.us(x3)],1)
        y3=torch.cat([self.mp2(x2),x3,self.us(x4)],1)
        y4=torch.cat([self.mp2(x3),x4],1)            
        x1=fun.relu(self.bn082(self.conv011(y1)))
        x2=fun.relu(self.bn162(self.conv021(y2)))
        x3=fun.relu(self.bn242(self.conv031(y3)))
        x4=fun.relu(self.bn322(self.conv041(y4)))
        y1=torch.cat([x1,self.us(x2)],1)
        y2=torch.cat([self.mp2(x1),x2,self.us(x3)],1)
        y3=torch.cat([self.mp2(x2),x3,self.us(x4)],1)
        y4=torch.cat([self.mp2(x3),x4],1)            
        x1=fun.relu(self.bn083(self.conv012(y1)))
        x2=fun.relu(self.bn163(self.conv022(y2)))
        x3=fun.relu(self.bn243(self.conv032(y3)))
        x4=fun.relu(self.bn323(self.conv042(y4)))
            
        y4=torch.cat([x3,self.us(x4)],1)
        y4=fun.relu(self.bn24(self.conv13(y4)))
        y3=torch.cat([x2,self.us(y4)],1)
        y3=fun.relu(self.bn16(self.conv12(y3)))
        y2=torch.cat([x1,self.us(y3)],1)
        y2=fun.relu(self.bn26(self.conv11(y2)))
        #y1=fun.relu(self.bn09(self.conv10(y2)))        
        y1=self.conv10(y2)
        return y1
model=Network2()