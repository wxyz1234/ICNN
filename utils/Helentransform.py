import torch
from torchvision import datasets,transforms
from skimage.util import random_noise
from torchvision.transforms import functional as TF
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DoNothing:
    def __call__(self, sample):
        return sample;
    
class ToPILImage:
        def __call__(self, sample):     
            #sample['label'] = np.uint8(sample['label'])
            sample['label'] = [TF.to_pil_image(sample['label'][i])
                  for i in range(sample['label'].shape[0])]
            return sample
class GaussianNoise:
    def __call__(self, sample):        
        sample['image'] = np.array(sample['image'], np.uint8)
        sample['image']=random_noise(sample['image'])
        sample['image'] = TF.to_pil_image(np.uint8(255 * sample['image']))  
        return sample
    
class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):                   
        '''
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, sample['image'].size)
        sample['image'] = TF.affine(sample['image'], *ret, resample=self.resample, fillcolor=self.fillcolor)
        sample['label']= [TF.affine(sample['label'][r], *ret, resample=self.resample, fillcolor=self.fillcolor) for r in range(len(sample['label']))]
        '''
        degree=random.uniform(self.degrees[0],self.degrees[1])          
        maxx=self.translate[0]*sample['image'].size[0];
        maxy=self.translate[1]*sample['image'].size[1];
        translate_range = (random.uniform(-maxx,maxx), random.uniform(-maxy,maxy));        
        scale_range = random.uniform(self.scale[0],self.scale[1]);          
        sample['image']=TF.affine(img=sample['image'],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample=Image.NEAREST)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.affine(img=sample['label'][i],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample=Image.NEAREST)                        
        return sample;
    
class Resize(transforms.Resize):
    def __call__(self, sample):             
        sample['image']=TF.resize(img=sample['image'],size=self.size,interpolation=Image.ANTIALIAS)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.resize(img=sample['label'][i],size=self.size,interpolation=Image.ANTIALIAS)
        return sample;
    
class ToTensor(transforms.ToTensor):
    def __call__(self, sample):        
        sample['image']=TF.to_tensor(sample['image'])    
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.to_tensor(sample['label'][i])         
        sample['label'][0]+=0.0001        
        sample['label']=torch.cat(tuple(sample['label']),0);#[L,64,64]
        return sample;    
    
class LabelUpdata:
    def __init__(self,choicesize):
        self.choicesize=choicesize;
    def __call__(self,sample):        
        '''
        lnum=len(sample['label']);
        sample['label']=torch.argmax(sample['label'],dim=0,keepdim=False);#[64,64]                
        sample['label']=sample['label'].unsqueeze(dim=0);#[1,64,64]        
        sample['label']=torch.zeros(lnum,self.choicesize,self.choicesize).scatter_(0, sample['label'], 1);#[L,64,64]  
        '''
        return sample;
'''
class ToPILImage():
    def __call__(self, sample):
        sample['image']=TF.to_pil_image(sample['image'])
        for i in range(len(sample['label'])):            
            sample['label'][i]=TF.to_pil_image(sample['label'][i])
        return sample;
'''
class HorizontalFlip():
    def __call__(self, sample):        
        sample['image']=TF.hflip(sample['image'])
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.hflip(sample['label'][i])
        return sample;