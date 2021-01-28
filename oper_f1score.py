import torch
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import functional as TF
import torch.optim as optim
from torch.autograd import Variable
from data.loaddata import test_data
from config import test_txt,batch_size,midout_path,image_path,label_path,midout_path2,output_path3
import sys
import os
from PIL import Image
import numpy as np
from data.loaddata import data_loader3
import shutil
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('TkAgg');
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
hists=[]
path_list=[]
path_num=0
unloader = transforms.ToPILImage()
trans=transforms.Compose([transforms.ToTensor()])
with open(test_txt) as f:                        
    lines=f.readlines()
    for line in lines:
        if (line.strip()==""):continue                                                                
        path=line.split(',')[1].strip()
        path_list.append(path);        
        #groundtruth labels
        path2=image_path+'/'+path+".jpg"
        image=Image.open(path2)
        label_list_gt=[]; 
        path2=label_path+'/'+path+'/'              
        #label_list_gt.append(image2)
        '''        
        image2=Image.open(path2+path+"_lbl00.png")   
        image2=image2.convert('L');            
        image2=trans(image2);                 
        label_list_gt.append(image2)   
        '''
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl0"+str(i)+".png")           
            label_list_gt.append(image2)             
        for i in range(len(label_list_gt)):
            label_list_gt[i]=np.array(label_list_gt[i]);          
        bg=255 - np.sum(label_list_gt, axis=0, keepdims=True)
        labels_list_gt = np.concatenate([bg, label_list_gt], axis=0)  # [L + 1, 64, 64]        
        labels_list_gt = np.uint8(labels_list_gt)             
        labels_list_gt = torch.from_numpy(labels_list_gt)                
        labels_list_gt=labels_list_gt.float()    
        labels_list_gt=torch.softmax(labels_list_gt, dim=0).argmax(dim=0, keepdim=False)         
        '''
        print(labels_list_gt.shape);        
        for i in range(labels_list_gt.shape[0]):
            for j in range(labels_list_gt.shape[1]):
                if labels_list_gt[i][j]!=0:
                    print(labels_list_gt[i][j]);
        input('wait');
        '''
        #prediction labels
        path2=image_path+'/'+path+".jpg"
        image=Image.open(path2)
        label_list_pre=[];            
        path2=output_path3+'/'+path+'/'
        for i in range(9):
            image2=Image.open(path2+path+"_lbl0"+str(i)+".jpg")   
            image2=image2.convert('L');            
            image2=trans(image2);
            label_list_pre.append(image2) 
            
            #plt.imshow(image2[0])
            #plt.show(block=True)    
        labels_list_pre = torch.cat(label_list_pre, dim=0)  
        #labels_list_pre[0]+=0.01;
        labels_list_pre=torch.softmax(labels_list_pre, dim=0).argmax(dim=0, keepdim=False)
        #print('label_pre.shape=',labels_list_pre.shape)        
        '''
        check1=0
        check2=0
        for i in labels_list_gt.reshape([-1]):
            if (i!=0):check1+=1;
        for i in labels_list_pre.reshape([-1]):
            if (i!=0):check2+=1;
        print("gt label num=",check1)
        print("pre label num=",check2)
        '''
        #merge  
        hist = np.bincount(9 * labels_list_gt.reshape([-1]) + labels_list_pre.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
    path_num=len(path_list);
#merge
hists_sum=np.sum(np.stack(hists, axis=0), axis=0)

for i in range(9):    
    for j in range(9):
        print(hists_sum[i][j],end=' ')
    print()
print();

#calc
f1=0.0;
tp=0;
tpfn=0;
tpfp=0;
for i in range(1,9):
    tp+=hists_sum[i][i].sum()
    tpfn+=hists_sum[i,:].sum()
    tpfp+=hists_sum[:,i].sum()    
'''
precision = tp / tpfp
recall = tp / tpfn
f1=2 * precision * recall / (precision + recall)
'''
f1=2*tp/(tpfn+tpfp)
print("tp is ",tp)
print("tp+fn is ",tpfn)
print("tp+fp is ",tpfp)
print("F1 Score is ",f1);
print();


#calc
f1=0.0;
tp=0;
tpfn=0;
tpfp=0;
for i in range(1,3):
    tp+=hists_sum[i][i].sum()
    tpfn+=hists_sum[i,:].sum()
    tpfp+=hists_sum[:,i].sum()
f1=2*tp/(tpfn+tpfp)
print("eyebrow tp is ",tp)
print("eyebrow tp+fn is ",tpfn)
print("eyebrow tp+fp is ",tpfp)
print("eyebrow F1 Score is ",f1);

f1=0.0;
tp=0;
tpfn=0;
tpfp=0;
for i in range(3,5):
    tp+=hists_sum[i][i].sum()
    tpfn+=hists_sum[i,:].sum()
    tpfp+=hists_sum[:,i].sum()
f1=2*tp/(tpfn+tpfp)
print("eye tp is ",tp)
print("eye tp+fn is ",tpfn)
print("eye tp+fp is ",tpfp)
print("eye F1 Score is ",f1);

f1=0.0;
tp=0;
tpfn=0;
tpfp=0;
for i in range(5,6):
    tp+=hists_sum[i][i].sum()
    tpfn+=hists_sum[i,:].sum()
    tpfp+=hists_sum[:,i].sum()
f1=2*tp/(tpfn+tpfp)
print("nose tp is ",tp)
print("nose tp+fn is ",tpfn)
print("nose tp+fp is ",tpfp)
print("nose F1 Score is ",f1);

f1=0.0;
tp=0;
tpfn=0;
tpfp=0;
for i in range(6,9):
    tp+=hists_sum[i][i].sum()
    tpfn+=hists_sum[i,:].sum()
    tpfp+=hists_sum[:,i].sum()
f1=2*tp/(tpfn+tpfp)
print("mouth tp is ",tp)
print("mouth tp+fn is ",tpfn)
print("mouth tp+fp is ",tpfp)
print("mouth F1 Score is ",f1);

print("Oper_F1score Finish!")