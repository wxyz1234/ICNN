import torch
import torchvision
from torchvision import datasets,transforms
import torch.optim as optim
from torch.autograd import Variable
from data.loaddata import data_data
from config import data_txt,batch_size,midout_path,image_path,midout_path2
import sys
import os
from PIL import Image
import numpy as np
from data.loaddata import data_loader3
import shutil

sizemouth=80;
sizeother=64;

def makeResult(part_name,model):
    global train_data,test_data,val_data;
    global path_list,path_num;
    unloader = transforms.ToPILImage()
    image_tran=[];    
        
    if (part_name=="mouth"):
        image_tran.append(transforms.Resize(size=(80, 80),interpolation=Image.NEAREST))
    else:
        image_tran.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
    if (part_name=="eye2" or part_name=="eyebrow2"):
        image_tran.append(transforms.RandomHorizontalFlip(p=1))
        #print('flip start')
    image_tran.append(transforms.ToTensor())
    image_tran=transforms.Compose(image_tran);
    data_data=data_loader3("data",batch_size,image_tran,None,part_name)
    k=0;
    for data in data_data.get_loader():
        if (use_gpu):
            data=data.cuda()        
        output=model(data)
        '''
        if (part_name=='mouth'):
            print('output')
            print(output.shape)
            print('output[0]')
            for i in range(80):                
                for j in range(80):                    
                    print(float(output[0][0][i][j]),end=' ')
                print()
            print('output[1]')
            for i in range(80):                
                for j in range(80):                    
                    print(float(output[0][1][i][j]),end=' ')
                print()
        '''
        for i in range(batch_size):
            path=midout_path2+'\\'+path_list[k]+'\\'+part_name;                
            data2=unloader(data[i].cpu())
            data2.save(path+"\\"+path_list[k]+".jpg",quality=100);
            image=output[i].cpu().clone();            
            image = torch.softmax(image, dim=0).argmax(dim=0, keepdim=False)
            image=image.unsqueeze(dim=0);              
            if (part_name=="mouth"):
                image=torch.zeros(4,sizemouth,sizemouth).scatter_(0, image, 255)  
            else:
                image=torch.zeros(2,sizeother,sizeother).scatter_(0, image, 255)  
            if not os.path.exists(path):
                os.makedirs(path);
            if (part_name=="mouth"):
                for j in range(4):
                    image2=unloader(np.uint8(image[j].numpy()))
                    image2.save(path+"\\"+path_list[k]+"_lbl0"+str(j)+".jpg",quality=100);
            else:
                for j in range(2):
                    image2=unloader(np.uint8(image[j].numpy()))
                    if (part_name=="eye2" or part_name=="eyebrow2"):
                        image2=image2.transpose(Image.FLIP_LEFT_RIGHT)
                        #print('flip end')
                    image2.save(path+"\\"+path_list[k]+"_lbl0"+str(j)+".jpg",quality=100);
            k+=1
            if (k>=path_num):break
    print("Oper_stage2 part "+part_name+" Finish!")
 
           
path_list=[]
path_num=0
with open(data_txt) as f:                        
    lines=f.readlines()
    for line in lines:
        if (line.strip()==""):continue                                                                
        path=line.split(',')[1].strip()
        path_list.append(path);
        if not os.path.exists(midout_path2+"\\"+path):
            os.makedirs(midout_path2+"\\"+path);  
        if not os.path.exists(midout_path2+"\\"+path+"\\image"):
            os.makedirs(midout_path2+"\\"+path+"\\image");  
        shutil.copy(midout_path+"\\"+path+"\\a.npy",midout_path2+"\\"+path+"\\a.npy")
        shutil.copy(midout_path+"\\"+path+"\\image\\"+path+".jpg",midout_path2+"\\"+path+"\\image\\"+path+".jpg")
    path_num=len(path_list);
use_gpu = torch.cuda.is_available()
print("use_gpu=",use_gpu)

#f=open("ImageOutput.txt",'w');
#f2=open("ImageOutput2.txt",'w');

model=torch.load("./BestNet2_eye", map_location='cpu')
if (use_gpu):
    model=model.cuda() 
makeResult("eye1",model);
makeResult("eye2",model);

model=torch.load("./BestNet2_eyebrow", map_location='cpu')
if (use_gpu):
    model=model.cuda() 
makeResult("eyebrow1",model);
makeResult("eyebrow2",model);

model=torch.load("./BestNet2_nose", map_location='cpu')
if (use_gpu):
    model=model.cuda() 
makeResult("nose",model);

model=torch.load("./BestNet2_mouth", map_location='cpu')
if (use_gpu):
    model=model.cuda() 
makeResult("mouth",model);

print("Oper_stage2 Finish!")