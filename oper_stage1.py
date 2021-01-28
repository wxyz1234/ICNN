import torch
import torchvision
from torchvision import datasets,transforms
import torch.optim as optim
from torch.autograd import Variable
from data.loaddata import test_data
from config import test_txt,batch_size,midout_path,image_path
import sys
import os
from PIL import Image
import numpy as np
'''
    Label 00: background    
    Label 01: left eyebrow
    Label 02: right eyebrow
    Label 03: left eye
    Label 04: right eye
    Label 05: nose
    Label 06: upper lip
    Label 07: inner mouth
    Label 08: lower lip
'''
#sizemouth=80;
#sizeother=64;
sizemouth=96;
sizeother=96

def makeResult(part_name,output,imagename):    
    image=Image.open(image_path+'\\'+imagename+".jpg")
    if not os.path.exists(path+"\\"+part_name):
        os.makedirs(path+"\\"+part_name); 
    image.save(path+"\\image\\"+imagename+".jpg",quality=100);
    if (part_name=="eye1"):
        part_num=3;
    if (part_name=="eye2"):
        part_num=4;
    if (part_name=="eyebrow1"):
        part_num=1;
    if (part_name=="eyebrow2"):
        part_num=2;
    if (part_name=="nose"):
        part_num=5;
    if (part_name=="mouth"):
        part_num=6;
        for x1 in range(64):
            for y1 in range(64):
                output[6][x1][y1]+=output[7][x1][y1]+output[8][x1][y1];
    xsum=0;
    ysum=0;
    numsum=0;     
    for x1 in range(64):
        for y1 in range(64):            
            if (output[part_num][y1][x1]>200):
                xsum+=x1;
                ysum+=y1;
                numsum+=1;    
    x1=image.size[0];
    y1=image.size[1];
    if (numsum==0):numsum+=1    
    xsum=int((xsum/numsum+1)/64*x1-1)
    ysum=int((ysum/numsum+1)/64*y1-1)
    if (part_name=="mouth"):
        xsum=max(0,xsum-sizemouth/2);
        ysum=max(0,ysum-sizemouth/2);
        xsum=min(xsum,image.size[0]-sizemouth)
        ysum=min(ysum,image.size[1]-sizemouth)
        image2=image.crop((xsum,ysum,xsum+sizemouth,ysum+sizemouth));
    else:
        xsum=max(0,xsum-sizeother/2);
        ysum=max(0,ysum-sizeother/2);
        xsum=min(xsum,image.size[0]-sizeother)
        ysum=min(ysum,image.size[1]-sizeother)    
        image2=image.crop((xsum,ysum,xsum+sizeother,ysum+sizeother));
    '''
    xsum=int((xsum/numsum+1)/64*x1-1)
    ysum=int((ysum/numsum+1)/64*y1-1)
    xsum=max(0,xsum-48);
    ysum=max(0,ysum-48);
    xsum=min(xsum,image.size[0]-96)
    ysum=min(ysum,image.size[1]-96)    
    image2=image.crop((xsum,ysum,xsum+96,ysum+96));
    '''
    image2.save(path+"\\"+part_name+"\\"+imagename+".jpg",quality=100);
    return (xsum,ysum);

path_list=[]
path_num=0
with open(test_txt) as f:                        
    lines=f.readlines()
    for line in lines:
        if (line.strip()==""):continue                                                                
        path=line.split(',')[1].strip()
        path_list.append(path);
    path_num=len(path_list);
k=0
use_gpu = torch.cuda.is_available()
model=torch.load("./BestNet",map_location="cpu") 
print("use_gpu=",use_gpu)
if (use_gpu):
    model=model.cuda() 
unloader = transforms.ToPILImage()
for data,target in test_data.get_loader():
    if (use_gpu):
        data=data.cuda()
        target=target.cuda()
    data,target=Variable(data),Variable(target)    
    output=model(data)    
    for i in range(batch_size):        
        path=midout_path+'\\'+path_list[k];                    
        image=output[i].cpu().clone();             
        image = torch.softmax(image, dim=0).argmax(dim=0, keepdim=False)               
        image=image.unsqueeze(dim=0);   
        image=torch.zeros(9,64,64).scatter_(0, image, 255)  
        if not os.path.exists(path+"\\labels"):
            os.makedirs(path+"\\labels"); 
        for j in range(9): 
            image2=unloader(np.uint8(image[j].numpy()))
            image2.save(path+"\\labels\\"+path_list[k]+"_lbl0"+str(j)+".jpg",quality=100);
        if not os.path.exists(path):
            os.makedirs(path);      
        if not os.path.exists(path+"\\image"):
            os.makedirs(path+"\\image"); 
        a=np.zeros((6,2),dtype=int)
        a[0]=makeResult("eye1",image,path_list[k]);
        a[1]=makeResult("eye2",image,path_list[k]);
        a[2]=makeResult("eyebrow1",image,path_list[k]);
        a[3]=makeResult("eyebrow2",image,path_list[k]);
        a[4]=makeResult("nose",image,path_list[k]);
        a[5]=makeResult("mouth",image,path_list[k]);   
        np.save(path+"\\"+"a.npy",a)
        k+=1
        if (k>=path_num):break
print("Oper_stage1 Finish!")