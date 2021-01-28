import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import sys
import os
from skimage import io
import matplotlib.pyplot as plt
import pylab
from config import batch_size,output_path,epoch_num,loss_image_path
from model.model import model
#from models.model_1 import FaceModel, Stage2FaceModel
#model=FaceModel()
from data.loaddata import train_data,test_data,val_data
#from torch.autograd import Variable
from PIL import Image
import numpy as np
#import matplotlib;matplotlib.use('TkAgg');
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

bestloss=1000000
bestf1=0
def train(epoch):        
    for batch_idx,sample in enumerate(train_data.get_loader()): 
        '''        
        for i in range(batch_size):
            image=sample['image'][i].cpu().clone();                
            image=transforms.ToPILImage()(image).convert('RGB')
            plt.imshow(image);
            plt.show(block=True);
        '''
        if (use_gpu):
            sample['image']=sample['image'].to(device)
            for i in range(len(sample['label'])):
                sample['label'][i]=sample['label'][i].to(device)                  
        optimizer.zero_grad()
        output=model(sample['image'])                                              
        labels=sample['label'].argmax(dim=1,keepdim=False)
        '''
        f = open("out.txt", "w")   
        for i in range(64):            
            for j in range(64):
                print(int(labels[0][i][j]),end=' ',file=f);
            print("",file=f);        
        input('pause');
        '''
        if (use_gpu):
            labels=labels.to(device);
        loss=fun.cross_entropy(output,labels)
        if (batch_idx%250==0):
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
    for data,target in val_data.get_loader():
        if (use_gpu):
            data=data.to(device)
            target=target.to(device)
        #data,target=Variable(data),Variable(target)   
        
        output=model(data)  
        target=torch.softmax(target, dim=1).argmax(dim=1, keepdim=False);  
        if (use_gpu):
            test_loss+=fun.cross_entropy(output,target,size_average=False).to(device).data
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
            sample['image']=sample['image'].to(device)
            for i in range(len(sample['label'])):
                sample['label'][i]=sample['label'][i].to(device)
        data=Variable(sample['image'])        
        target=torch.cat(tuple(sample['label']),1)
        target=target.argmax(dim=1,keepdim=False)
        target=Variable(target)                 
        
        output=model(data)  
        if (use_gpu):
            test_loss+=fun.cross_entropy(output,target,size_average=False).to(device).data
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
    test_loss/=len(test_data.get_loader().dataset)
    print('\nTest set: {} Cases，Average loss: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),test_loss))
    print("stage1 tp=",tp)
    print("stage1 tpfp=",tpfp)
    print("stage1 tpfn=",tpfn)    
    print('\nTest set: {} Cases，F1 Score: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),f1score))
    loss_list.append(test_loss.data.cpu().numpy());
    f1_list.append(f1score);
    if (f1score>bestf1):
        bestf1=f1score
        print("Best data Stage1 Updata\n");
        torch.save(model,"./BestNet")            
    '''    
    if (test_loss<bestloss):
        bestloss=test_loss
        torch.save(model,"./BestNet")           
    '''    
def printoutput1(print_data):
    model=torch.load("./BestNet",map_location="cpu")
    if (use_gpu):
        model=model.to(device)
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
            data=data.to(device)
            target=target.to(device)
        output=model(data)
        output2=output.cpu().clone();                 
        output2 = torch.softmax(output2, dim=1).argmax(dim=1, keepdim=False)               
        image2=target.cpu().clone()    
        image2 = torch.softmax(image2, dim=1).argmax(dim=1, keepdim=False)               
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
def makeplt():
    loss_list=np.load(loss_image_path+'\\loss_list_stage1.npy')
    loss_list=loss_list.tolist();
    f1_list=np.load(loss_image_path+'\\f1_list_stage1.npy')
    f1_list=f1_list.tolist();
    x_list=np.load(loss_image_path+'\\x_list_stage1.npy')
    x_list=x_list.tolist();
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x_list, loss_list,'r',label="loss")    
    ax2 = ax1.twinx()
    ax2.plot(x_list, f1_list,'b',label="f1_score")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("f1_score")    
    ax1.legend(loc=2);
    ax2.legend(loc=4);
    
    plt.savefig(loss_image_path+'\\loss_stage1.jpg');    
loss_list=[];
f1_list=[];
x_list=[];
print("use_gpu=",use_gpu)
if (use_gpu):
    model=model.to(device)
#model=torch.load("./Netdata")    
#optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
optimizer=optim.Adam(model.parameters(),lr=0.001) 
scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)       
for epoch in range(epoch_num):
    x_list.append(epoch);
    train(epoch)
    scheduler.step()
    test()
torch.save(model,"./Netdata")
printoutput1(test_data)  
x_list_stage1=np.array(x_list)
np.save(loss_image_path+'\\x_list_stage1.npy',x_list_stage1) 
f1_list_stage1=np.array(f1_list)
np.save(loss_image_path+'\\f1_list_stage1.npy',f1_list_stage1) 
loss_list_stage1=np.array(loss_list)
np.save(loss_image_path+'\\loss_list_stage1.npy',loss_list_stage1) 
makeplt();

