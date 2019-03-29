import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import re

from model.train import GhibliModel
from PIL import Image

imsize = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator:
    def __init__(self):
        self.model = GhibliModel()
        self.state_dict = torch.load("./model/results/model2.pth", map_location = 'cpu')
        for k in list(self.state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del self.state_dict[k]
        self.model.net.load_state_dict(self.state_dict, strict=False)
        self.model.net.to(device)
    
    def generate(self, request):
        '''
        This method reads the file uploaded from the Flask application POST request, 
        and performs a generation using the Ghibli Model. 
        '''
        f = request.files['image']
        f=Image.open(f)
        loader = transforms.Compose([    
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))])
        content = loader(f).unsqueeze(0)
        content.to(device, torch.float)
        output = self.model.net(content)
        '''
        output[:,0,:,:] += 103.939
        output[:,1,:,:] += 116.779
        output[:,2,:,:] +=123.68
        '''
        output = output.detach()
        output = output.numpy().transpose(0,2,3,1).reshape(256,256,3)
        output = np.clip(output, 0, 255).astype('uint8')
        output_img = Image.fromarray(output)
        output_img.save("./static/output.jpg")
        return output_img
        
        
	