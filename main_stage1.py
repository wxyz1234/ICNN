import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import sys
import os
from skimage import io
os.environ['CUDA_VISIBLE_DEVICES']='4, 5'
import matplotlib.pyplot as plt
import pylab
from config import batch_size,output_path
from model.model import model
#from models.model_1 import FaceModel, Stage2FaceModel
#model=FaceModel()
from data.loaddata import train_data,test_data,val_data
#from torch.autograd import Variable
from PIL import Image
import numpy as np
use_gpu = torch.cuda.is_available()
bestloss=1000000
bestf1=0
def train(epoch):        
    for batch_idx,sample in enumerate(train_data.get_loader()):    
        if (use_gpu):
            sample['image']=sample['image'].cuda()
            for i in range(len(sample['label'])):
                sample['label'][i]=sample['label'][i].cuda()
        optimizer.zero_grad()
        output=model(sample['image'])                      
        #labels=np.concatenate(tuple(sample['label']),axis=1)
        #labels=np.argmax(labels,axis=1)        
        labels=torch.cat(tuple(sample['label']),1)
        labels=labels.argmax(dim=1,keepdim=False)
        loss=fun.cross_entropy(output,labels)
        if (batch_idx%50==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample['image']), len(train_data.get_loader().dataset),
                100. * batch_idx / len(train_data.get_loader()),loss.data))                    
        loss.backward()
        optimizer.step()        
        #params = list(model.named_parameters())
        #print("params len:",params.__len__())
        #print("params[4]:",params[4])
        #print("params[3]:",params[3])
def test():
    global bestloss,bestf1
    test_loss=0
    correct=0
    hists=[]
    for data,target in test_data.get_loader():
        if (use_gpu):
            data=data.cuda()
            target=target.cuda()
        #data,target=Variable(data),Variable(target)   
        
        output=model(data)  
        if (use_gpu):
            test_loss+=fun.cross_entropy(output,target,size_average=False).cuda().data
        else:
            test_loss+=fun.cross_entropy(output,target,size_average=False).data        
        image=output.cpu().clone()
        image = torch.softmax(image, dim=1).argmax(dim=1, keepdim=False)             
        output=target.cpu().clone();
        hist = np.bincount(9 * output.reshape([-1]) + image.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);              
    '''
    for sample in test_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].cuda()
            for i in range(len(sample['label'])):
                sample['label'][i]=sample['label'][i].cuda()
        data=Variable(sample['image'])        
        target=torch.cat(tuple(sample['label']),1)
        target=target.argmax(dim=1,keepdim=False)
        target=Variable(target)                 
        
        output=model(data)  
        if (use_gpu):
            test_loss+=fun.cross_entropy(output,target,size_average=False).cuda().data
        else:
            test_loss+=fun.cross_entropy(output,target,size_average=False).data        
        image=output.cpu().clone()
        image = torch.softmax(image, dim=1).argmax(dim=1, keepdim=False)             
        output=target.cpu().clone();
        hist = np.bincount(9 * output.reshape([-1]) + image.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);                          
    '''
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()    
    f1score=2*tp/(tpfn+tpfp)
    if (f1score>bestf1):
        bestf1=f1score
        print("Best data Stage1 Updata");
        torch.save(model,"./BestNet")        
    print('\nTest set: {} Cases，F1 Score: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),f1score))
    '''
    test_loss/=len(test_data.get_loader().dataset)
    if (test_loss<bestloss):
        bestloss=test_loss
        torch.save(model,"./BestNet")       
    print('\nTest set: {} Cases，Average loss: {:.4f}\n'.format(len(test_data.get_loader().dataset),test_loss))
    '''
def printoutput1(print_data):
    model=torch.load("./BestNet",map_location="cpu")
    if (use_gpu):
        model=model.cuda()    
    unloader = transforms.ToPILImage()
    k=0;
    hists=[]                             
    '''
    for sample in print_data.get_loader():      
        data=sample['image']
        target=torch.cat(tuple(sample['label']),1)        
        target=target.argmax(dim=1,keepdim=False) 
    '''
    for data,target in print_data.get_loader():    
        if (use_gpu):
            data=data.cuda()
            target=target.cuda()
        output=model(data)
        output2=output.cpu().clone();                 
        output2 = torch.softmax(output2, dim=1).argmax(dim=1, keepdim=False)               
        image2=target.cpu().clone()        
        for i in range(batch_size):             
            k1=k%print_data.get_len();
            k2=k//print_data.get_len();
            path=output_path+'/'+print_data.get_namelist()[k1]+'_'+str(k2);   
            if not os.path.exists(path):
                os.makedirs(path);                
            image=data[i].cpu().clone();                
            image =unloader(image)
            image.save(path+'/'+print_data.get_namelist()[k1]+'.jpg',quality=100);    
                        
            image=output[i].cpu().clone();             
            image = torch.softmax(image, dim=0).argmax(dim=0, keepdim=False)               
            #image=target[i].cpu().clone();
            image=image.unsqueeze(dim=0);                                    
            image=torch.zeros(9,64,64).scatter_(0, image, 255)                
                        
            for j in range(9):                                
                image3=unloader(np.uint8(image[j].numpy()))                           
                image3.save(path+'/'+print_data.get_namelist()[k1]+'lbl0'+str(j)+'.jpg',quality=100);            
            k+=1
            if (k>=print_data.get_len()):break               
            #if (k>=print_data.get_len()):break            
        hist = np.bincount(9 * image2.reshape([-1]) + output2.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
        if (k>=print_data.get_len()):break
        #if (k>=print_data.get_len()):break
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()    
    f1score=2*tp/(tpfn+tpfp)
    print('Printoutput F1 Score: {:.4f}\n'.format(f1score))
    print("printoutput1 Finish");    
print("use_gpu=",use_gpu)
if (use_gpu):
    model=model.cuda()
#model=torch.load("./Netdata")    
#optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
optimizer=optim.Adam(model.parameters(),lr=0.001) 
for epoch in range(25):
    train(epoch)
    test()
torch.save(model,"./Netdata")
printoutput1(val_data)  

