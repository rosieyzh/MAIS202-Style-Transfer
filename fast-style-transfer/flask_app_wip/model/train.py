import torch.nn as nn
import matplotlib
import numpy as np
#matplotlib.use('TkAgg')
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import model.getmodel 
import model.processing_methods as PM

# Look to the path of your current working directory

ghibliImg =1
save_model_path = './results/model{}.pth'.format(ghibliImg)
save_optimizer_path = './results/optimizer{}.pth'.format(ghibliImg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256

def image_loader(batch_size=4):
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder('./data', transform=transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])), batch_size=batch_size, shuffle=True)
    return train_loader

class GhibliModel:
    def __init__(self, learning_rate=0.001):
        self.net = model.getmodel.Transformation().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
    
    def train_model(self, train_data, batch_size, content_weight, style_weight, randSeed, n_epochs=2, log_interval=100):
        np.random.seed(randSeed)
        torch.manual_seed(randSeed)

        vgg = model.getmodel.Vgg16(requires_grad=False).to(device)

        style = PM.load_image("../assets/style/ghibli_square_{}.png".format(ghibliImg))
        style = style.repeat(batch_size, 1, 1, 1).to(device)
        style = PM.normalization(style)
        print(style)
        features_recon = vgg(PM.normalization(style))
        style_recon = [PM.gram_matrix(x) for x in features_recon]
        for epoch in range(1, n_epochs + 1):
            self.net.train()
            style_score = 0.
            content_score = 0.
            count = 0
            for batch_idx, (data, _) in enumerate(train_data):
                curbatch = len(data)
                count += curbatch
                self.optimizer.zero_grad()
                data = data.to(device)
                y = self.net(data)
                
                y = PM.normalization(y)
                data = PM.normalization(data)
                
                features_y = vgg(y)
                features_data = vgg(data)
                
                content_loss = content_weight*self.mse_loss(features_y.relu2_2, features_data.relu2_2)
                style_loss = 0.
                for feat_y, gram_s in zip(features_y, style_recon):
                    gram_y = PM.gram_matrix(feat_y)
                    style_loss += self.mse_loss(gram_y, gram_s[:curbatch,  :, :])
                style_loss *= style_weight
                
                total_loss = content_loss + style_loss
                total_loss.backward()
                self.optimizer.step()
                
                content_score += content_loss.item()
                style_score += style_loss.item()
                
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tContent Loss: {:.6f}\tStyle Loss: {:.6f}\tTotal: {:.6f}'.format(
						epoch, batch_idx * len(data), len(train_data.dataset),
						100. * batch_idx / len(train_data), content_score / (batch_idx + 1),
                                  style_score / (batch_idx + 1),
                                  (content_score+style_score) / (batch_idx + 1)))
                    torch.save(self.net.state_dict(), save_model_path)
                    torch.save(self.optimizer.state_dict(), save_optimizer_path)
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), save_model_path)
        print("\nDone, model was successfully saved!")

if __name__ == "__main__" :
	model = GhibliModel()
	random_seed = 42
	torch.manual_seed(random_seed)

	# Load the data. 
	train_loader = image_loader()

	# Train the model
	model.train_model(train_loader, 4, 500, 1e7, random_seed)
	
	
	

