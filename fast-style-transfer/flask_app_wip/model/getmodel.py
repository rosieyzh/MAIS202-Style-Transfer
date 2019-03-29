import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import collections

class Transformation(nn.Module):
    def __init__(self):
        super(Transformation, self).__init__()
        #padding to maintain final dimensions at the end
        self.pad = nn.ReflectionPad2d(40)
        self.conv_1 = ConvLayer(3, 32, kernel=9, stride=1)
        self.conv_2 = ConvLayer(32, 64, kernel=3, stride=2)
        self.conv_3 = ConvLayer(64, 128, kernel=3, stride=2)
        self.resid_1 = ResidualLayer(128)
        self.resid_2 = ResidualLayer(128)
        self.resid_3 = ResidualLayer(128)
        self.resid_4 = ResidualLayer(128)
        self.resid_5 = ResidualLayer(128)
        self.conv_4 = UpSampConvLayer(128, 64, kernel=3, stride=1, upsample = 2)
        self.conv_5 = UpSampConvLayer(64, 32, kernel=3, stride=1, upsample = 2)
        self.conv_6 = ConvLayer(32, 3, kernel=9, stride=1)
        self.relu = nn.ReLU()
    def forward(self, X):
        out = self.relu(self.conv_1(self.pad(X)))
        out = self.relu(self.conv_2(out))
        out = self.relu(self.conv_3(out))
        out = self.resid_1(out)
        out = self.resid_2(out)
        out = self.resid_3(out)
        out = self.resid_4(out)
        out = self.resid_5(out)
        out = self.relu(self.conv_4(out))
        out = self.relu(self.conv_5(out))
        out = self.relu(self.conv_6(out))
        return out



class ConvLayer(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel, stride):
        super(ConvLayer, self).__init__()
        #to maintain input/output spatial dimension
        reflection_padding = kernel // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.constpad = nn.ConstantPad2d((1,0,1,0), 0)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.conv(self.reflection_pad(x))
        out = self.norm(out)
        return out

class UpSampConvLayer(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel, stride, upsample = None):
        super(UpSampConvLayer, self).__init__()
        #to maintain input/output spatial dimension
        reflection_padding = kernel // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.conv(self.reflection_pad(x))
        out = self.norm(out)
        return out

class ResidualLayer(nn.Module):
    def __init__ (self, channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.batch1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.batch2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        residual = F.interpolate(residual, size=(out.shape[2], out.shape[3]), mode = 'nearest')
        out = out + residual
        return out


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = models.vgg16(pretrained=True).features
        self.conv_1 = nn.Sequential()
        self.conv_2 = nn.Sequential()
        self.conv_3 = nn.Sequential()
        self.conv_4 = nn.Sequential()
        for layer in range(23):
            if layer < 4:
                self.conv_1.add_module(str(layer), pretrained_vgg[layer])
            elif layer < 9:
                self.conv_2.add_module(str(layer), pretrained_vgg[layer])
            elif layer < 16:
                self.conv_3.add_module(str(layer), pretrained_vgg[layer])
            else:
                self.conv_4.add_module(str(layer), pretrained_vgg[layer])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv_1(x)
        in_relu_1_2 = x
        x = self.conv_2(x)
        in_relu_2_2 = x
        x = self.conv_3(x)
        in_relu_3_3 = x
        x = self.conv_4(x)
        in_relu_4_3 = x
        vgg_style_out = collections.namedtuple("Vgg_style_out", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        output = vgg_style_out(in_relu_1_2, in_relu_2_2, in_relu_3_3, in_relu_4_3)
        return output

    
