'''
Generates multiple images for style transfer.
'''
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import random

import style_transfer_methods as st
style_files = [ "./data/style/ghibli_square_4.png"]
style_masks = [
  "./data/style/ghibli_square_1_mask.png"]
content_files = [
  "./data/content/real_square_2_3.jpg"]
content_masks = [
  "./data/masks/real_square_1_2_mask.jpg"]

#using image masks?
use_masks = True
num_mask_colours = 3

device = torch.device("cuda")
cnn = models.vgg19(pretrained = True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.485, 0.456, 0.406]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
display = transforms.ToPILImage()
#variable to rename files
i=17

for fileIndex in range(len(style_files)):
	style_file = style_files[fileIndex]
	style_mask_file = style_masks[fileIndex]
	content_file = content_files[fileIndex]
	style_mask_file = style_masks[fileIndex]
	imsize = 512

	loader = transforms.Compose([
		transforms.Resize(imsize),
		transforms.ToTensor()])
	
	style_img = st.load_image(style_file)
	content_img = st.load_image(content_file)

	if use_masks == True & num_mask_colours > 1:
		style_mask = st.load_image(style_mask_file)
		content_mask = st.load_image(content_mask_file)
	
	assert style_img.size() == content_img.size()
	input_img_1 = content_img.clone()
	#input_img = st.get_random_noise_image(512, 512)
	#input_img = transforms.ToTensor()(input_img).unsqueeze(0) 
	#input_img_1 = input_img.to(device, torch.float)
	
	output = st.style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img_1, i)
	
	plt.figure()
	st.imshow(output, title='Output Image')

	plt.show()
	plt.savefig("./data/results/output_{}_input_img_1_300.png".format(i), format="png")
	i += 1
