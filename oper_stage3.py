import torch
import torchvision
from torchvision import datasets,transforms
import torch.optim as optim
from torch.autograd import Variable
from data.loaddata import data_data
from config import data_txt,batch_size,midout_path,image_path,midout_path2,output_path3
import sys
import os
from PIL import Image
import numpy as np
from data.loaddata import data_loader3
import shutil

#image_size=64;
#image_size_mouth=80;
image_size=96;
image_size_mouth=96;

def change(ans,part_name,path):        
    image=Image.open(midout_path2+"\\"+path+'\\'+part_name+'\\'+path+'_lbl00.jpg')    
    image=image.resize((image_size,image_size),Image.NEAREST)
    image=image.convert('L')
    if (part_name=='eye1'):k=0;
    if (part_name=='eye2'):k=1;
    if (part_name=='eyebrow1'):k=2;
    if (part_name=='eyebrow2'):k=3;
    if (part_name=='nose'):k=4;
    if (part_name=='mouth'):k=5;
    '''
    if (part_name=='eye1' or part_name=='eyebrow1'):
        print(path,' ',part_name);
        print(a[k][1],' ',a[k][0]);
    '''  
    if (part_name=='mouth'):
        for i in range(image_size_mouth):
            for j in range(image_size_mouth):            
                ans[0][a[k][1]+j][a[k][0]+i]=min(ans[0][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
    else:
        for i in range(image_size):
            for j in range(image_size):            
                ans[0][a[k][1]+j][a[k][0]+i]=min(ans[0][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
    image=Image.open(midout_path2+"\\"+path+'\\'+part_name+'\\'+path+'_lbl01.jpg')
    image=image.resize((image_size,image_size),Image.NEAREST)
    image=image.convert('L')
    if (part_name=='eye1'):
        for i in range(image_size):
            for j in range(image_size):
                ans[3][a[k][1]+j][a[k][0]+i]=max(ans[3][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
    if (part_name=='eye2'):
        for i in range(image_size):
            for j in range(image_size):
                ans[4][a[k][1]+j][a[k][0]+i]=max(ans[4][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))                
    if (part_name=='eyebrow1'):
        for i in range(image_size):
            for j in range(image_size):
                ans[1][a[k][1]+j][a[k][0]+i]=max(ans[1][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
    if (part_name=='eyebrow2'):
        for i in range(image_size):
            for j in range(image_size):
                ans[2][a[k][1]+j][a[k][0]+i]=max(ans[2][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))                                
    if (part_name=='nose'):
        for i in range(image_size):
            for j in range(image_size):
                ans[5][a[k][1]+j][a[k][0]+i]=max(ans[5][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
    if (part_name=='mouth'):
        for i in range(image_size_mouth):
            for j in range(image_size_mouth):
                ans[6][a[k][1]+j][a[k][0]+i]=max(ans[6][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
        image=Image.open(midout_path2+"\\"+path+'\\'+part_name+'\\'+path+'_lbl02.jpg')
        image=image.resize((image_size_mouth,image_size_mouth),Image.NEAREST)
        image=image.convert('L')
        for i in range(image_size_mouth):
            for j in range(image_size_mouth):
                ans[7][a[k][1]+j][a[k][0]+i]=max(ans[7][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
        image=Image.open(midout_path2+"\\"+path+'\\'+part_name+'\\'+path+'_lbl03.jpg')
        image=image.resize((image_size_mouth,image_size_mouth),Image.NEAREST)
        image=image.convert('L')
        for i in range(image_size_mouth):
            for j in range(image_size_mouth):
                ans[8][a[k][1]+j][a[k][0]+i]=max(ans[8][a[k][1]+j][a[k][0]+i],image.getpixel((i,j)))
            
path_list=[]
path_num=0
unloader = transforms.ToPILImage()
with open(data_txt) as f:                        
    lines=f.readlines()
    for line in lines:
        if (line.strip()==""):continue                                                                
        path=line.split(',')[1].strip()
        path_list.append(path);
        if not os.path.exists(output_path3+"\\"+path):
            os.makedirs(output_path3+"\\"+path);                
        shutil.copy(midout_path2+"\\"+path+"\\image\\"+path+".jpg",output_path3+"\\"+path+"\\"+path+".jpg")
        image=Image.open(output_path3+"\\"+path+"\\"+path+".jpg")
        a=np.load(midout_path2+"\\"+path+"\\a.npy")        
        x1=image.size[0]
        y1=image.size[1]
        ans=np.zeros((9,y1,x1));        
        ans[0]=255;        
        change(ans,'eye1',path);
        change(ans,'eye2',path);
        change(ans,'eyebrow1',path);
        change(ans,'eyebrow2',path);
        change(ans,'nose',path);
        change(ans,'mouth',path);
        for i in range(9):
            image=unloader(np.uint8(ans[i]))
            image.save(output_path3+"\\"+path+"\\"+path+"_lbl0"+str(i)+".jpg",quality=100);
    path_num=len(path_list);
print("Oper_stage3 Finish!")