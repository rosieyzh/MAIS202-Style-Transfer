import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import random

from sklearn.cluster import KMeans

device = torch.device("cuda")
imsize = 512
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])
display = transforms.ToPILImage()

def load_image(image_file):
    image = Image.open(image_file)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def load_mask(mask_file):
    mask = Image.open(mask_file)
    new_mask = np.array(transforms.Resize(imsize)(mask))
    return new_mask

#takes our tensors and converts them to viewable images in pyplot
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = display(image)
    if title is not None:
        plt.title(title)
    plt.imshow(image)

#generates a random white noise image for input
def get_random_noise_image(width, height):
    im = Image.new("RGB", (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    im.putdata(list(random_grid))
    return im

#Define two loss functions

class ContentLoss(nn.Module):
    '''
        Content Loss defined as a 'distance' between intermediate outputs and content image
        Mean squared error of two feature maps
        
    '''
    def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()
            
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
        
#gram matrix used for computing style loss
def gram_matrix(input):
    #gram matrix (feature maps multiplied by its transpose) needed for calculating style loss
    #normalized because large N => large gram matrix entries
    #which means the initial layers before pooling will matter more than the deeper layers
    
    '''
        b: batch size
        f: number of feature maps
        (w,h): dimensions of a feature map (i.e. N = wh)
    '''
    b,f,w,h = input.size()
    features = input.view(b*f, w*h) #need to resize feature map into a K*N matrix
    
    G = torch.mm(features, features.t())
    
    #normalize values by dividing number of elements
    return G.div(b*f*w*h)

def guided_gram_matrix(input, guidance_channel):
    #gram matrix (feature maps multiplied by its transpose) needed for calculating style loss
    #normalized because large N => large gram matrix entries
    #which means the initial layers before pooling will matter more than the deeper layers
    
    '''
        b: batch size
        f: number of feature maps
        (w,h): dimensions of a feature map (i.e. N = wh)
    '''
    b,f,w,h = input.size()
    features = input.view(b*f, w*h) #need to resize feature map into a K*N matrix
    features = features*(guidance_channel.view(b*f, w*h))
    
    G = torch.mm(features, features.t())
    
    #normalize values by dividing number of elements
    return G.div(b*f*w*h)

class StyleLoss(nn.Module):
    '''
        Style Loss similarly uses mean squared error between gram matrices of style and intermediate inputs
    '''
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__ (self, mean, std):
        super(Normalization, self).__init__()
        '''
            need input image as [C, 1, 1]
            image Tensor has shape [B, C, H, W]
            B: Batch size
            C: Number of channels
            H: Height
            W: Width
        '''
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    
    def forward(self, image):
        return (image - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_colourful_masks (style_mask, content_mask, imsize, num_mask_colours):
    content_mask = content_mask.reshape([imsize*imsize, -1])
    style_mask = style_mask.reshape([imsize*imsize, -1])

    assert content_mask.shape == style_mask.shape
    
    kmeans = KMeans(n_clusters=num_mask_colours, random_state=0).fit(style_mask)
    content_labels = kmeans.predict(content_mask.astype(np.float32))
    content_labels = content_labels.reshape([imsize, imsize])
    style_labels = kmeans.predict(style_mask.astype(np.float32))
    style_labels = style_labels.reshape([imsize, imsize])
    
    content_masks = []
    style_masks = []
    
    for colour in range(num_mask_colours):
        content_masks.append((content_labels == colour).astype(np.float32))
        style_masks.append((style_labels == colour).astype(np.float32))
    #***CHANGE DEVICE TO CUDA***
    content_stack = np.stack(content_masks)
    style_stack = np.stack(style_masks)
    content_stack = content_stack[np.newaxis, :, :, :]
    style_stack = content_stack[np.newaxis, :, :, :]
    content_masks_tensor = torch.tensor(content_stack, device = 'cpu')
    style_masks_tensor = torch.tensor(style_stack, device = 'cpu')
    return content_masks_tensor, style_masks_tensor

def get_mask_model (cnn, mask, mask_layers = mask_layers_default):
    cnn = copy.deepcopy(cnn)
    
    model = nn.Sequential()
    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.MaxPool2d):
            i+=1
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            model.add_module(name, layer)
    return model

def compute_guidance_channel (masks, downsample_type = 'simple'):
    '''
    Propagating input guidance channel to form guidance channel at each layer.
    "Simple" down-sampling to the size of the feature map - i.e. we only apply the pooling layers 
    '''
    features={}
    model = get_mask_model(cnn, masks)
    for layer in model.children():
            output = layer(masks)
            features[layer] = output
    return features

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default, style_layers = style_layers_default):
    cnn = copy.deepcopy(cnn)
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    content_losses=[]
    style_losses=[]
    
    model = nn.Sequential(normalization)
    
    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+= 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'batchnorm_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
        model.add_module(name, layer)
        
        if name in content_layers:
            #adding content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            #adding style loss
            target_feature = model(style_img)
            target_feature.detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
    model = model[:(i+1)]
    
    #replace max pooling with avg pooling
    '''
    model.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
    model.pool_4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
    '''
    return model, style_losses, content_losses

#define optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

#Run style transfer
#number of steps and weight to apply to differences for style and content loss
steps=300
s_weight = 1000000
c_weight = 1

def style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, i, num_steps=steps, style_weight=s_weight, content_weight = c_weight):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    
    #arrays to graph style loss and content loss vs steps
    s_loss_points = []
    c_loss_points = []
    steps=[]
    
    #tracking number of steps
    run = [0]
    while run[0] <= num_steps:
        
        def closure():
            input_img.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            
            for s in style_losses:
                style_score += s.loss
            for c in content_losses:
                content_score += c.loss
            
            style_score *= style_weight
            content_score *= content_weight
            
            #loss function is our weighted sum between style and content, defined in paper
            loss = style_score + content_score
            #minimize this loss
            loss.backward()
            
            run[0]+= 1
            
            #printing to track progress, add points to plot
            if run[0] % 25 == 0:
                print("run {}:".format(run))
                print("Style Loss: {:4f}, Content Loss: {:4f}".format(style_score.item(), content_score.item()))
                s_loss_points.append(style_score.item())
                c_loss_points.append(content_score.item())
                steps.append(run[0])
        
            return style_score + content_score
        
        optimizer.step(closure)
    #ensures our normalized image pixel values are still between 0 and 1
    input_img.data.clamp_(0,1)
    
    #plot graph
    plt.figure()
    plt.plot(steps, s_loss_points, c='r')
    plt.plot(steps, c_loss_points, c='b')
    plt.title("Style and Content Loss")
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    plt.savefig("./data/results/graph_{}_avg_pooling_300_1000000_whitenoise_max.png".format(i), format="png")
    
    return input_img

