import time,datetime
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import sys
import os
from PIL import Image
#os.environ['CUDA_VISIBLE_DEVICES']='4, 5'
import matplotlib.pyplot as plt
import pylab
from config import batch_size,output_path,output_path2
from model.model import Network2
from models.model_1 import FaceModel, Stage2FaceModel
from data.loaddata import data_loader2,Augmentation_part2,data_loader2_Aug,TransFromAug
#from torch.autograd import Variable
from PIL import Image
import numpy as np
#import matplotlib;matplotlib.use('TkAgg');
use_gpu = torch.cuda.is_available()
bestloss=1000000
bestf1=0

train_eye=True
train_eyebrow=True
train_nose=True
train_mouth=True

def train(part_name,epoch,model,optimizer):        
    print("use_gpu=",use_gpu);    
    #part1_time=0;    
    #part2_time=0;    
    #part3_time=0;    
    #prev_time=time.time();
    for batch_idx,sample in enumerate(train_data.get_loader()):                            
        #now_time=time.time();
        #part3_time+=now_time-prev_time;        
        #prev_time=now_time;
        
        if (use_gpu):
            sample['image']=sample['image'].cuda()
            sample['label']=sample['label'].cuda()            
        data=sample['image']                        
        target=sample['label']    
        
        optimizer.zero_grad()            
        #data,target=Variable(data),Variable(target)    
        output=model(data) 
        '''
        print(data.size());             
        print(target.size());
        print(output.size()); 
        image=data[0];        
        image=transforms.ToPILImage()(image).convert('RGB')
        image.save("out.jpg");
        f = open("out.txt", "w")   
        for i in range(64):            
            for j in range(64):
                print(int(target[0][0][i][j]),end=' ',file=f);
            print("",file=f);
        print("",file=f);
        for i in range(64):            
            for j in range(64):
                print(int(target[0][1][i][j]),end=' ',file=f);
            print("",file=f);                             
        os.system('pause');
        '''
        #now_time=time.time();
        #part1_time+=now_time-prev_time;        
        #prev_time=now_time;
        
        '''        
        if (part_name=="eye"):                                    
            for p in range(target.shape[0]):         
                if (target[p,4].sum()>target[p,3].sum()):
                    target[p,3]=target[p,4]                        
            target=target[:,[0,3]]            
        if (part_name=="eyebrow"):            
            for p in range(target.shape[0]):
                if (target[p,2].sum()>target[p,1].sum()):
                    target[p,1]=target[p,2]
            target=target[:,[0,1]]            
        if (part_name=="nose"):                                          
            target=target[:,[0,5]]            
        if (part_name=="mouth"):            
            target=target[:,[0,6,7,8]]            
        '''    
        target=torch.softmax(target, dim=1).argmax(dim=1, keepdim=False);                    
        loss=fun.cross_entropy(output,target)
                
        loss.backward()
        optimizer.step()
        
        #now_time=time.time();
        #part2_time+=now_time-prev_time;        
        #prev_time=now_time;
                    
        if (batch_idx%250==0):
            print(part_name+' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.get_loader().dataset),
                100. * batch_idx / len(train_data.get_loader()),loss.data))        
            #print("part1_time=",part1_time);     
            #print("part2_time=",part2_time);     
            #print("part3_time=",part3_time);     
def test(part_name,model,optimizer):
    global bestloss,bestf1
    test_loss=0    
    hists=[]
    for data,target in test_data.get_loader():
        #data,target=Variable(data),Variable(target)  
        target=target.unsqueeze(dim=1);     
        if (part_name=="mouth"):
            target=torch.zeros(target.shape[0],9,80,80).scatter_(1, target, 255);
        else:
            target=torch.zeros(target.shape[0],9,64,64).scatter_(1, target, 255);  
        if (use_gpu):
            data=data.cuda()
            target=target.cuda()          
        output=model(data)      
        if (part_name=="eye"):                        
            #for k in range(4):target[k][3]+=target[k][4];  
            for p in range(target.shape[0]):         
                if (target[p,4].sum()>target[p,3].sum()):
                    target[p,3]=target[p,4]
            target=target[:,[0,3]]                                           
        if (part_name=="eyebrow"):            
            #for k in range(4):target[k][1]+=target[k][2];
            for p in range(target.shape[0]):
                if (target[p,2].sum()>target[p,1].sum()):
                    target[p,1]=target[p,2]
            target=target[:,[0,1]]
        if (part_name=="nose"):
            target=target[:,[0,5]]            
        if (part_name=="mouth"):
            target=target[:,[0,6,7,8]]   
        target[:,0]+=0.01;
        target=torch.softmax(target, dim=1).argmax(dim=1, keepdim=False);
        test_loss=fun.cross_entropy(output,target)              
        image=output.clone()        
        image = torch.softmax(image, dim=1).argmax(dim=1, keepdim=False)              
        output=target.clone();  
        '''
        if (part_name=="mouth"):
            for k in range(4):            
                for i in range(80):
                    for j in range(80):
                        if ((output[k][i][j]==image[k][i][j])&(output[k][i][j]!=0)):                            
                            tp=tp+1
                        if ((output[k][i][j]!=image[k][i][j])&(image[k][i][j]!=0)):
                            fp=fp+1                    
                        if ((output[k][i][j]!=image[k][i][j])&(output[k][i][j]!=0)):
                            fn=fn+1            
        else:
            for k in range(4):            
                for i in range(64):
                    for j in range(64):
                        if ((output[k][i][j]==image[k][i][j])&(output[k][i][j]!=0)):                            
                            tp=tp+1
                        if ((output[k][i][j]!=image[k][i][j])&(image[k][i][j]!=0)):                       
                            fp=fp+1
                        if ((output[k][i][j]!=image[k][i][j])&(output[k][i][j]!=0)):                            
                            fn=fn+1          
        '''
        '''
        f = open("out.txt", "w")   
        for i in range(64):
            for j in range(64):
                print(int(output[0][i][j]),end=' ',file=f)
            print(file=f)
        print(file=f)
        for i in range(64):
            for j in range(64):
                print(int(output[1][i][j]),end=' ',file=f)
            print(file=f)  
        print("f output");    
        f.close();
        '''
        hist = np.bincount(9 * output.cpu().reshape([-1]) + image.cpu().reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;    
    '''
    for i in range(9):
        for j in range(9):
            print(hists_sum[i][j],end=' ')
        print()
    '''
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()    
    f1score=2*tp/(tpfn+tpfp)
    if (f1score>bestf1):
        bestf1=f1score
        print("Best data "+part_name+" Updata");
        torch.save(model,"./BestNet2_"+part_name)   
    print("tp=",tp)
    print("tpfp=",tpfp)
    print("tpfn=",tpfn)    
    print('\nTest set: {} Cases，F1 Score: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),f1score))
    '''
    test_loss/=len(test_data.get_loader().dataset)    
    if (test_loss<bestloss):
        bestloss=test_loss
        torch.save(model,"./BestNet2_"+part_name)
    print('\nTest set: {} Cases，Average loss: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),test_loss))
    '''
def printoutput1(part_name,print_data):
    if (use_gpu):
        model=torch.load("./Netdata2_"+part_name)     
    else:
        model=torch.load("./BestNet2_"+part_name,map_location="cpu")    
    unloader = transforms.ToPILImage()
    k=0;
    kcheck=1;
    hists=[]
    if ((part_name=="eye")or(part_name=="eyebrow")):
        kcheck=2;        
    '''
    for sample in print_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].cuda()
            for i in range(len(sample['label'])):
                sample['label'][i]=sample['label'][i].cuda()
        data=sample['image']        
        target=torch.cat(tuple(sample['label']),1) 
    '''
    for data,target in print_data.get_loader():             
        target=target.unsqueeze(dim=1);                   
        if (part_name=="mouth"):             
            target=torch.zeros(target.shape[0],9,80,80).scatter_(1, target, 255);
        else:
            target=torch.zeros(target.shape[0],9,64,64).scatter_(1, target, 255);    
        if (use_gpu):
            data=data.cuda()
            target=target.cuda()
        if (part_name=="eye"):                        
            #for k in range(4):target[k][3]+=target[k][4]; 
            for p in range(target.shape[0]):         
                if (target[p,4].sum()>target[p,3].sum()):
                    target[p,3]=target[p,4]
            target=target[:,[0,3]]                                  
        if (part_name=="eyebrow"):            
            #for k in range(4):target[k][1]+=target[k][2];
            for p in range(target.shape[0]):
                if (target[p,2].sum()>target[p,1].sum()):
                    target[p,1]=target[p,2]
            target=target[:,[0,1]]            
        if (part_name=="nose"):
            target=target[:,[0,5]]
        if (part_name=="mouth"):
            target=target[:,[0,6,7,8]]
                
        output=model(data)        
        for i in range(batch_size):
            k1=k%(print_data.get_len()*kcheck);
            k2=k//(print_data.get_len()*kcheck);
            if (part_name=="eye"):
                if (k1%2==0):
                    part_name2="eye1"
                else:
                    part_name2="eye2"
                path=output_path2+'/'+part_name+'/'+print_data.get_namelist()[k1//2]+'_'+str(k2)+'/'+part_name2;
            if (part_name=="eyebrow"):
                if (k1%2==0):
                    part_name2="eyebrow1"
                else:
                    part_name2="eyebrow2"
                path=output_path2+'/'+part_name+'/'+print_data.get_namelist()[k1//2]+'_'+str(k2)+'/'+part_name2;
            if (part_name=="nose"):
                path=output_path2+'/'+part_name+'/'+print_data.get_namelist()[k1]+'_'+str(k2)+'/nose';
            if (part_name=="mouth"):
                path=output_path2+'/'+part_name+'/'+print_data.get_namelist()[k1]+'_'+str(k2)+'/mouth';
            if not os.path.exists(path):
                os.makedirs(path);
            image=data[i].cpu().clone();                
            image =unloader(image)            
            image.save(path+'/'+print_data.get_namelist()[k1//2]+'.jpg');                
            
            image=output[i].cpu().clone();              
            #image=target[i].cpu().clone();
            image = torch.softmax(image, dim=0).argmax(dim=0, keepdim=False)                           
            image=image.unsqueeze(dim=0);            
            if (part_name=="mouth"):
                image=torch.zeros(9,80,80).scatter_(0, image, 255);
            else:
                image=torch.zeros(9,64,64).scatter_(0, image, 255);  
            if (part_name=='eye'):
                for j in range(2):
                    image3=unloader(np.uint8(image[j].numpy()))           
                    #plt.imshow(image3)
                    #plt.show(block=True)
                    #print(np.uint8(image[j].numpy())[32])                
                    #image3=unloader(image[j])
                    image3.save(path+'/'+print_data.get_namelist()[k1//2]+'lbl0'+str(j)+'.jpg',quality=100);
            if (part_name=='eyebrow'):
                for j in range(2):
                    image3=unloader(np.uint8(image[j].numpy()))           
                    #plt.imshow(image3)
                    #plt.show(block=True)
                    #print(np.uint8(image[j].numpy())[32])                
                    #image3=unloader(image[j])
                    image3.save(path+'/'+print_data.get_namelist()[k1//2]+'lbl0'+str(j)+'.jpg',quality=100);                
            if (part_name=='mouth'):
                for j in range(4):
                    image3=unloader(np.uint8(image[j].numpy()))           
                    #plt.imshow(image3)
                    #plt.show(block=True)
                    #print(np.uint8(image[j].numpy())[32])                
                    #image3=unloader(image[j])
                    image3.save(path+'/'+print_data.get_namelist()[k1]+'lbl0'+str(j)+'.jpg',quality=100);                
            if (part_name=='nose'):
                for j in range(2):
                    image3=unloader(np.uint8(image[j].numpy()))           
                    #plt.imshow(image3)
                    #plt.show(block=True)
                    #print(np.uint8(image[j].numpy())[32])                
                    #image3=unloader(image[j])
                    image3.save(path+'/'+print_data.get_namelist()[k1]+'lbl0'+str(j)+'.jpg',quality=100);                
            k+=1
            if (k>=print_data.get_len()*kcheck):break
            #if (k>=print_data.get_len()*kcheck*5):break
        target[:,0]+=1        
        output=torch.softmax(output, dim=1).argmax(dim=1, keepdim=False);
        target=torch.softmax(target, dim=1).argmax(dim=1, keepdim=False);
        '''        
        f1=open('./output.txt','w')
        f2=open('./output2.txt','w')
        for i2 in range(64):
            for j2 in range(64):
                print(int(output[1][i2][j2]),end=' ',file=f1)
            print(file=f1)
        for i2 in range(64):
            for j2 in range(64):
                print(int(target[1][i2][j2]),end=' ',file=f2)
            print(file=f2)
        f1.close()
        f2.close()
        sys.pause()        
        '''
        hist = np.bincount(9 * output.cpu().reshape([-1]) + target.cpu().reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
        #if (k>=print_data.get_len()*kcheck):break
        if (k>=print_data.get_len()*kcheck*5):break
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
    print(part_name," tp=",tp)
    print(part_name," tpfp=",tpfp)
    print(part_name," tpfn=",tpfn)    
    print('\n',part_name,' F1 Score: {:.4f}\n'.format(f1score))
    print(part_name+" printoutput1 Finish");    
def train_model(part_name):
    global train_data,test_data,val_data,model;
    if (part_name=="mouth"):
        label_tran=[];
        label_tran.append(transforms.Resize(size=(80, 80),interpolation=Image.NEAREST))
        label_tran.append(transforms.ToTensor())
        label_tran=transforms.Compose(label_tran);
        image_tran=[];
        image_tran.append(transforms.Resize(size=(80, 80),interpolation=Image.NEAREST))
        image_tran.append(transforms.ToTensor())
        image_tran=transforms.Compose(image_tran);
        image_tran2=[];
        image_tran2.append(transforms.Resize(size=(80, 80),interpolation=Image.NEAREST))
        image_tran2.append(transforms.RandomHorizontalFlip(p=1))
        image_tran2.append(transforms.ToTensor())
        image_tran2=transforms.Compose(image_tran2);        
        trans=TransFromAug(Augmentation_part2().augmentation,80).trans;
    else:
        label_tran=[];
        label_tran.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
        label_tran.append(transforms.ToTensor())
        label_tran=transforms.Compose(label_tran);
        image_tran=[];
        image_tran.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
        image_tran.append(transforms.ToTensor())
        image_tran=transforms.Compose(image_tran);
        image_tran2=[];
        image_tran2.append(transforms.Resize(size=(64, 64),interpolation=Image.NEAREST))
        image_tran2.append(transforms.RandomHorizontalFlip(p=1))
        image_tran2.append(transforms.ToTensor())
        image_tran2=transforms.Compose(image_tran2);
        trans=TransFromAug(Augmentation_part2().augmentation,64).trans;
    if (part_name=="eye"):
        model=Network2(output_maps=2);        
        train_data=data_loader2_Aug("train",batch_size,trans,"eye");
        test_data=data_loader2("test",batch_size,image_tran,image_tran2,label_tran,"eye");
        val_data=data_loader2("val",batch_size,image_tran,image_tran2,label_tran,"eye");
        #val_data=data_loader2_Aug("val",batch_size,trans,"eye");
    if (part_name=="eyebrow"):
        model=Network2(output_maps=2);        
        train_data=data_loader2_Aug("train",batch_size,trans,"eyebrow");
        test_data=data_loader2("test",batch_size,image_tran,image_tran2,label_tran,"eyebrow");
        val_data=data_loader2("val",batch_size,image_tran,image_tran2,label_tran,"eyebrow");
    if (part_name=="nose"):
        model=Network2(output_maps=2);        
        train_data=data_loader2_Aug("train",batch_size,trans,"nose");
        test_data=data_loader2("test",batch_size,image_tran,image_tran2,label_tran,"nose");
        val_data=data_loader2("val",batch_size,image_tran,image_tran2,label_tran,"nose");
    if (part_name=="mouth"):
        model=Network2(output_maps=4);        
        train_data=data_loader2_Aug("train",batch_size,trans,"mouth");
        test_data=data_loader2("test",batch_size,image_tran,image_tran2,label_tran,"mouth");
        val_data=data_loader2("val",batch_size,image_tran,image_tran2,label_tran,"mouth");    
    if (use_gpu):
        model=model.cuda()
    #model=torch.load("./Netdata2_"+part_name)                    
    optimizer=optim.Adam(model.parameters(),lr=0.0025)
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)    
    for epoch in range(25):
        train(part_name,epoch,model,optimizer)     
        scheduler.step()
        test(part_name,model,optimizer)        
    torch.save(model,"./Netdata2_"+part_name)                                                      
    printoutput1(part_name,val_data)   
print("use_gpu=",use_gpu)
if (train_eye):
    train_model("eye");
if (train_eyebrow):
    train_model("eyebrow");
if (train_nose):
    train_model("nose");
if (train_mouth):
    train_model("mouth")
