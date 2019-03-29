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


#Uploading files

style_file = "./data/style/ghibli_square_4.png"
content_file = "./data/content/real_square_4.JPG"

#feeding in images of 224 x 224 px
imsize = 512

#running on cpu for now
device = torch.device("cuda")

#resize images and convert to tensor
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

def load_image(image_file):
    image = Image.open(image_file)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = load_image(style_file)
content_img = load_image(content_file)

assert style_img.size() == content_img.size()

#Displaying files

display = transforms.ToPILImage()

#takes our tensors and converts them to viewable images in pyplot
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = display(image)
    if title is not None:
        plt.title(title)
    plt.imshow(image)


#plt.figure()
#imshow(style_img, title="Style Image")

#plt.figure()
#imshow(content_img, title="Content Image")

#input image is our content image
input_img_1 = content_img.clone()

#plt.figure()
#imshow(input_img_1, title='Input Image')

#alternatively start image with random white noise
input_img_2 = torch.randn(content_img.data.size(), device=device)
#plt.figure()
#input_img_2.show()


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
   
#Using pretrained 19-layer VGG as proposed in the paper
cnn = models.vgg19(pretrained = True).features.to(device).eval()

#vgg is trained on channels normalized by mean (0.485, 0.456, 0,406) and std (0.229, 0.224, 0.225)
#***NORMALIZE IMAGES BEFORE SENDING THEM THROUGH
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.485, 0.456, 0.406]).to(device)

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

#retrieve model for style transfer from pretrained network
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

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

def style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=steps, style_weight=s_weight, content_weight = c_weight):
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
    plt.savefig("./data/results/graph_4_avg_pooling_300_1000000_input_img_1_max.png", format="png")
    
    return input_img

#running this bad boy
output = style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img_1)

#displays output image
plt.figure()
imshow(output, title='Output Image')

plt.show()
plt.savefig("./data/results/output_4_300_1000000_input_img_1_max.png", format="png")
