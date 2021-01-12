import time,datetime
import torch
import random
from torchvision import datasets,transforms
from torch.utils import data
from PIL import Image
from torchvision.transforms import functional as TF
import numpy as np
import copy 
import os
import sys
sys.path.append("..")
from skimage import io
from config import train_txt,val_txt,test_txt,data_txt,image_path,label_path,batch_size,facail_part_path,midout_path
from utils.Helentransform import GaussianNoise,RandomAffine,Resize,ToTensor,ToPILImage,HorizontalFlip
'''
    Label 00: background
    Label 01: face skin (excluding ears and neck)
    Label 02: left eyebrow
    Label 03: right eyebrow
    Label 04: left eye
    Label 05: right eye
    Label 06: nose
    Label 07: upper lip
    Label 08: inner mouth
    Label 09: lower lip
    Label 10: hair
'''
            
class Augmentation:
    def __init__(self):        

        degree = 15
        translate_range = (0.1,0.1);
        scale_range = (0.9, 1.2);                
        self.augmentation=[];
        self.augmentation.append([]);
        self.augmentation.append([GaussianNoise(),
                                    RandomAffine(degrees=degree, translate=translate_range,scale=scale_range),
                                    transforms.Compose([GaussianNoise(),RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)])]);
                
        self.augmentation.append([RandomAffine(degrees=degree, translate=(0,0), scale=(1,1)),
                                    RandomAffine(degrees=0, translate=translate_range, scale=(1,1)),
                                    RandomAffine(degrees=0, translate=(0,0), scale=scale_range)]);
  
        self.augmentation.append([RandomAffine(degrees=0, translate=translate_range, scale=scale_range),
                                    RandomAffine(degrees=degree, translate=(0,0), scale=scale_range),
                                    RandomAffine(degrees=degree, translate=translate_range, scale=(1,1))]);
                
        self.augmentation.append([RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)]);        
               
class Augmentation_part2:
    def __init__(self):                
        
        degree = 15
        translate_range = (0.1,0.1);
        scale_range = (0.9, 1.2);                
        self.augmentation=[];
        self.augmentation.append([]);
        self.augmentation.append([GaussianNoise(),
                                    RandomAffine(degrees=degree, translate=translate_range,scale=scale_range),
                                    transforms.Compose([GaussianNoise(),RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)])]);
                
        self.augmentation.append([RandomAffine(degrees=degree, translate=(0,0), scale=(1,1)),
                                    RandomAffine(degrees=0, translate=translate_range, scale=(1,1)),
                                    RandomAffine(degrees=0, translate=(0,0), scale=scale_range)]);
  
        self.augmentation.append([RandomAffine(degrees=0, translate=translate_range, scale=scale_range),
                                    RandomAffine(degrees=degree, translate=(0,0), scale=scale_range),
                                    RandomAffine(degrees=degree, translate=translate_range, scale=(1,1))]);
                
        self.augmentation.append([RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)]);        

class TransFromAug:
    def __init__(self,augmentation,size):
        self.trans=[];        
        self.trans.append(transforms.Compose([                
                Resize(size=(size, size),interpolation=Image.NEAREST),                
                ToTensor()
            ]));
        self.trans.append(transforms.Compose([                
                transforms.RandomChoice(augmentation[1]),
                Resize(size=(size, size),interpolation=Image.NEAREST),                
                ToTensor()
            ]));
        self.trans.append(transforms.Compose([                
                transforms.RandomChoice(augmentation[2]),
                Resize(size=(size, size),interpolation=Image.NEAREST),                
                ToTensor()
            ]));
        self.trans.append(transforms.Compose([                
                transforms.RandomChoice(augmentation[3]),
                Resize(size=(size, size),interpolation=Image.NEAREST),                
                ToTensor()
            ]));
        self.trans.append(transforms.Compose([                
                transforms.RandomChoice(augmentation[4]),
                Resize(size=(size, size),interpolation=Image.NEAREST),                
                ToTensor()
            ]));        


class data_loader:
    def __init__(self,mode,batch_size,image_trans,label_trans):
        self._path_list=[]
        self._num=0;
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);
        self._data_loader=get_loader(mode,batch_size,image_trans,label_trans);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;                
class data_loader_Aug:
    def __init__(self,mode,batch_size,trans):
        self._path_list=[]
        self._num=0;
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);
        self._data_loader=get_loader_Aug(mode,batch_size,trans);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;      
class data_loader2:
    def __init__(self,mode,batch_size,image_trans,image_trans2,label_trans,face_part):
        self._path_list=[]
        self._num=0;
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);        
        self._data_loader=get_loader2(mode,face_part,batch_size,image_trans,image_trans2,label_trans);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;
class data_loader2_Aug:
    def __init__(self,mode,batch_size,trans,face_part):
        self._path_list=[]
        self._num=0;
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);        
        self._data_loader=get_loader2_Aug(mode,face_part,batch_size,trans);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;      
class data_loader3:
    def __init__(self,mode,batch_size,image_trans,label_trans,face_part):
        self._path_list=[]
        self._num=0;
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);        
        self._data_loader=get_loader3(mode,face_part,batch_size,image_trans,label_trans);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;       

class Helen(data.Dataset):
    def __init__(self,image_transform=None,label_transform=None,mode='train'):
        self.image_transform=image_transform
        self.label_transform=label_transform
        self.path_list=[]
        self.num=0;
        self.mode=mode
        self.preprocess_data(mode);
    def __len__(self):
        return self.num;
    def __getitem__(self,index):
        path=self.path_list[index]
        path2=image_path+'/'+path+".jpg"
        image=Image.open(path2)
        label_list=[]; 
        label_list.append(Image.new('L', (image.size[0],image.size[1]),color=1))               
        path2=label_path+'/'+path+'/'
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png")   
            image2=image2.convert('L');            
            label_list.append(image2) 
        if self.image_transform!=None:
            image=self.image_transform(image)
        if self.label_transform!=None:
            for i,j in enumerate(label_list):
                label_list[i]=self.label_transform(j)          
        labels_list = torch.cat(label_list, dim=0)         
        labels_list=torch.argmax(labels_list,dim=0)        
        return image,labels_list
    def preprocess_data(self,mode):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                self.path_list.append(path);
            self.num=len(self.path_list);
        print("Preprocess the {} data, it has {} images".format(mode, self.num))
class Helen_Aug(data.Dataset):
    def __init__(self,trans=None,mode='train'):
        self.trans=trans        
        self.path_list=[]
        self.listnum=0;
        self.num=0;
        self.mode=mode
        self.preprocess_data(mode);    
    def __len__(self):
        return self.num;
    def __getitem__(self,index): 
        if (self.trans!=None):
            image_trans=self.trans[index//self.listnum]
        else:
            image_trans=None
        index2=index%self.listnum;        
        path=self.path_list[index2]
        path2=image_path+'/'+path+".jpg"        
        image=Image.open(path2)
        label=[]; 
        label.append(Image.new('L', (image.size[0],image.size[1]),color=1))               
        path2=label_path+'/'+path+'/'
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png")   
            image2=image2.convert('L');            
            label.append(image2) 
        sample={"image":image,"label":label,"index":index}
        if (image_trans!=None):
            sample=image_trans(sample);        
        return sample
    def preprocess_data(self,mode):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                self.path_list.append(path);
            self.listnum=len(self.path_list);
            self.num=self.listnum*5;
        print("Preprocess the {} data, it has {} images".format(mode, self.num))        
        
class Helen2(data.Dataset):
    def __init__(self,image_transform=None,image_transform2=None,label_transform=None,mode='train',face_part='eye'):
        self.image_transform=image_transform
        self.image_transform2=image_transform2
        self.label_transform=label_transform
        self.path_list=[]
        self.num=0;
        self.mode=mode
        self.face_part=face_part;
        self.preprocess_data(mode,face_part);        
    def __len__(self):
        return self.num;
    def __getitem__(self,index):
        path=self.path_list[index][0]
        part=self.path_list[index][1]        
        path2=facail_part_path+'/'+part+'/images/'+path+".jpg"        
        image=Image.open(path2)
        label_list=[]; 
        label_list.append(Image.new('L', (image.size[0],image.size[1]),color=1))               
        path2=facail_part_path+'/'+part+'/labels/'+path+'/'
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png")   
            image2=image2.convert('L'); 
            if (i==3 or i==5):
                image2=image2.transpose(Image.FLIP_LEFT_RIGHT)             
            label_list.append(image2) 
            '''
            x3=0;
            print("size is ",image2.size[0]," ",image2.size[1])
            for x1 in range(image2.size[0]):                
                for x2 in range(image2.size[1]):
                    if image2.getpixel((x1,x2))!=0:
                        #print("x1=",x1,"x2=",x2,"pixel=",image2.getpixel((x1,x2)))               
                        x3+=1;
            print("label"+str(i)+"Num of pixel is "+str(x3)+" Before resize")
            '''
        if (part=='eye2' or part=='eyebrow2'):
            if self.image_transform2!=None:
                image=self.image_transform2(image)
        else:
            if self.image_transform!=None:
                image=self.image_transform(image)
        if self.label_transform!=None:
            for i,j in enumerate(label_list):
                label_list[i]=self.label_transform(j)          
        labels_list = torch.cat(label_list, dim=0)         
        '''
        for i in range(9):
            x3=0;
            for x1 in range(64):                
                for x2 in range(64):       
                    if (labels_list[i][x1][x2]!=0):
                        x3+=1;
            print(print("label"+str(i)+"Num of pixel is "+str(x3)+" After resize"));        
        '''
        labels_list=torch.argmax(labels_list,dim=0)                              
        return image,labels_list
    def preprocess_data(self,mode,face_part):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                if (face_part=='eye'):
                    self.path_list.append((path,'eye1'));
                    self.path_list.append((path,'eye2'));
                if (face_part=='eyebrow'):
                    self.path_list.append((path,'eyebrow1'));
                    self.path_list.append((path,'eyebrow2'));
                if (face_part=='nose'):
                    self.path_list.append((path,'nose'));
                if (face_part=='mouth'):
                    self.path_list.append((path,'mouth'));
            self.num=len(self.path_list);
        print("Preprocess the {} data, it has {} images".format(mode, self.num))   

class Helen2_Aug(data.Dataset):
    def __init__(self,trans=None,mode='train',face_part='eye'):
        self.trans=trans
        self.path_list=[]
        self.num=0;
        self.listnum=0;
        self.mode=mode
        self.face_part=face_part;
        self.preprocess_data(mode,face_part);        
    def __len__(self):
        return self.num;
    def __getitem__(self,index):   
        #prev_time=time.time();        
        
        index2=index%self.listnum;
        path=self.path_list[index2][0]
        part=self.path_list[index2][1]        
        path2=facail_part_path+'/'+part+'/images/'+path+".jpg" 
        if (self.trans!=None):
            image_trans=self.trans[index//self.listnum]
        else:
            image_trans=None;            
        image=Image.open(path2)
        label=[]; 
        label.append(Image.new('L', (image.size[0],image.size[1]),color=1))               
        path2=facail_part_path+'/'+part+'/labels/'+path+'/'        
        '''
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png")   
            image2=image2.convert('L');            
            label.append(image2);        
        sample={'image':image,'label':label,'index':index,'part':self.face_part};
        if (image_trans!=None):
            sample=image_trans(sample);
        sample['label']=torch.cat(tuple(sample['label']),0);
        #print("sample['label'].size=",sample['label'].size());
        if (self.face_part=="eye"):                                                
            if (sample['label'][4].sum()>sample['label'][3].sum()):
                sample['label'][3]=sample['label'][4]                        
            sample['label']=sample['label'][[0,3]]
        if (self.face_part=="eyebrow"):            
            if (sample['label'][2].sum()>sample['label'][1].sum()):
                sample['label'][1]=sample['label'][2]   
            sample['label']=sample['label'][[0,1]]
        if (self.face_part=="nose"):                                          
            sample['label']=sample['label'][[0,5]]
        if (self.face_part=="mouth"):            
            sample['label']=sample['label'][[0,6,7,8]]
        '''
        if (part=="eye1"):  
            for i in range(4,5):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L');            
                label.append(image2);    
        if (part=="eye2"):  
            for i in range(5,6):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L');            
                image2=image2.transpose(Image.FLIP_LEFT_RIGHT)
                label.append(image2);     
        if (part=="eyebrow1"):  
            for i in range(2,3):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L');            
                label.append(image2);            
        if (part=="eyebrow2"):  
            for i in range(3,4):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L'); 
                image2=image2.transpose(Image.FLIP_LEFT_RIGHT)
                label.append(image2);          
        if (part=="nose"):  
            for i in range(6,7):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L');            
                label.append(image2);            
        if (part=="mouth"):  
            for i in range(7,10):
                image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png").convert('L');
                #image2=image2.convert('L');            
                label.append(image2);
        #print("sample['label'].size=",sample['label'].size());
        #now_time=time.time();
        #print("getitem time part1:",now_time-prev_time);
        #pre_time=now_time;
                
        sample={'image':image,'label':label,'index':index,'part':self.face_part};
        if (image_trans!=None):
            sample=image_trans(sample);
        sample['label']=torch.cat(tuple(sample['label']),0);                    
            
        #now_time=time.time();
        #print("getitem time part2:",now_time-prev_time);
        #pre_time=now_time;        
        return sample
    def preprocess_data(self,mode,face_part):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                if (face_part=='eye'):
                    self.path_list.append((path,'eye1'));
                    self.path_list.append((path,'eye2'));
                if (face_part=='eyebrow'):
                    self.path_list.append((path,'eyebrow1'));
                    self.path_list.append((path,'eyebrow2'));
                if (face_part=='nose'):
                    self.path_list.append((path,'nose'));
                if (face_part=='mouth'):
                    self.path_list.append((path,'mouth'));
            self.listnum=len(self.path_list);
            self.num=self.listnum*5;
        print("Preprocess the {} data, it has {} images".format(mode, self.num))   
  
class Helen3(data.Dataset):
    def __init__(self,image_transform=None,label_transform=None,mode='train',face_part='eye'):
        self.image_transform=image_transform
        self.label_transform=label_transform
        self.path_list=[]
        self.num=0;
        self.mode=mode
        self.face_part=face_part;
        self.preprocess_data(mode,face_part);        
    def __len__(self):
        return self.num;
    def __getitem__(self,index):
        path=self.path_list[index][0]
        part=self.path_list[index][1]        
        path2=midout_path+'/'+path+'/'+part+'/'+path+".jpg"        
        image=Image.open(path2)                
        if self.image_transform!=None:
            image=self.image_transform(image)        
        return image
    def preprocess_data(self,mode,face_part):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt
        if (mode=='data'):file_path=data_txt
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                self.path_list.append((path,face_part));                
            self.num=len(self.path_list);
        print("Preprocess the {} data, it has {} images".format(mode, self.num))  

def get_loader(mode="train", batch_size=4, image_transform=None, label_transform=None):
    dataset = Helen(image_transform=image_transform, label_transform=label_transform, mode=mode)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    return data_loader
def get_loader_Aug(mode="train", batch_size=4, trans=None):
    dataset = Helen_Aug(trans=trans, mode=mode)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    return data_loader
def get_loader2(mode="train",face_part="eye", batch_size=4, image_transform=None, image_transform2=None, label_transform=None):
    dataset = Helen2(image_transform=image_transform, image_transform2=image_transform2, label_transform=label_transform, mode=mode,face_part=face_part)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    return data_loader
def get_loader2_Aug(mode="train",face_part="eye", batch_size=4, trans=None):
    dataset = Helen2_Aug(trans=trans, mode=mode,face_part=face_part)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    return data_loader
def get_loader3(mode="train",face_part="eye", batch_size=4, image_transform=None, label_transform=None):
    dataset = Helen3(image_transform=image_transform, label_transform=label_transform, mode=mode,face_part=face_part)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    return data_loader

label_tran=[];
label_tran.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
label_tran.append(transforms.ToTensor())
label_trans=transforms.Compose(label_tran);
image_tran=[];
image_tran.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
image_tran.append(transforms.ToTensor())
image_trans=transforms.Compose(image_tran);
'''
train_data=data_loader("train",batch_size,image_trans,label_trans);
test_data=data_loader("test",batch_size,image_trans,label_trans);
val_data=data_loader("val",batch_size,image_trans,label_trans);
data_data=data_loader("data",batch_size,image_trans,label_trans);
'''
trans=TransFromAug(Augmentation().augmentation,64).trans;
train_data=data_loader_Aug("train",batch_size,trans);
test_data=data_loader("test",batch_size,image_trans,label_trans);
val_data=data_loader("val",batch_size,image_trans,label_trans);
#val_data=data_loader_Aug("val",batch_size,trans);
data_data=data_loader("data",batch_size,image_trans,label_trans);
print("Stage1 Load Data Finish")