from torchvision import datasets,transforms
from skimage.util import random_noise
from torchvision.transforms import functional as TF
import random
import numpy as np
from PIL import Image

class GaussianNoise:
    def __call__(self, sample):        
        sample['image'] = np.array(sample['image'], np.uint8)
        sample['image']=random_noise(sample['image'])
        sample['image'] = TF.to_pil_image(np.uint8(255 * sample['image']))  
        return sample
    
class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):           
        degree=random.uniform(self.degrees[0],self.degrees[1])
        translate_range = (random.uniform(-self.translate[0],self.translate[0]), random.uniform(-self.translate[1],self.translate[1]));
        scale_range = random.uniform(self.scale[0],self.scale[1]);           
        sample['image']=TF.affine(img=sample['image'],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.affine(img=sample['label'][i],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0)                
        return sample;
    
class Resize(transforms.Resize):
    def __call__(self, sample):             
        sample['image']=TF.resize(img=sample['image'],size=self.size,interpolation=Image.NEAREST)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.resize(img=sample['label'][i],size=self.size,interpolation=Image.NEAREST)
        return sample;
    
class ToTensor(transforms.ToTensor):
    def __call__(self, sample):        
        sample['image']=TF.to_tensor(sample['image'])        
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.to_tensor(sample['label'][i])                
        sample['label'][0]+=0.0001        
        return sample;    
    
class ToPILImage():
    def __call__(self, sample):
        sample['image']=TF.to_pil_image(sample['image'])
        for i in range(len(sample['label'])):            
            sample['label'][i]=TF.to_pil_image(sample['label'][i])
        return sample;

class HorizontalFlip():
    def __call__(self, sample):
        if (sample['part']=='eye2' or sample['part']=='eyebrow2'):
            sample['image']=TF.hflip(sample['image'])
            for i in range(len(sample['label'])):
                sample['label'][i]=TF.hflip(sample['label'][i])
        return sample;