import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256

def load_image(image_file):
    loader = transforms.Compose([    
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))])

    image = Image.open(image_file)
    image = loader(image).unsqueeze(0) 
    return image.to(device, torch.float)


def gram_matrix(input):
    #gram matrix (feature maps multiplied by its transpose) needed for calculating style loss
    #normalized because large N => large gram matrix entries
    #which means the initial layers before pooling will matter more than the deeper layers
    
    '''
        b: batch size
        f: number of feature maps
        (w,h): dimensions of a feature map (i.e. N = wh)
    '''
    (b,f,w,h) = input.size()
    features = input.view(b,f, w*h) #need to resize feature map into a K*N matrix
    features_transpose = features.transpose(1, 2)
    G = torch.bmm(features, features_transpose)
    
    #normalize values by dividing number of elements
    return G.div(f*h*w)

def normalization(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def unnormalization(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)
