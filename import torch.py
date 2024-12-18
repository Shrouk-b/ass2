import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
z_dim = 100  
lr = 0.0002
num_epochs = 50
img_size = 28 * 28 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class Generator(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_size),
            nn.Tanh()  
        )

    def forward(self, z):
        return self.model(z)
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )

    def forward(self, img):
        return self.model(img)

generator = Generator(z_dim, img_size).to(device)
discriminator = Discriminator(img_size).to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.view(-1, img_size).to(device)

        real_labels = torch.ones(real_imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)

        z = torch.randn(real_imgs.size(0), z_dim).to(device)
        fake_imgs = generator(z)
        d_real = discriminator(real_imgs)
        d_fake = discriminator(fake_imgs.detach())

        loss_D_real = criterion(d_real, real_labels)
        loss_D_fake = criterion(d_fake, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        z = torch.randn(real_imgs.size(0), z_dim).to(device)
        fake_imgs = generator(z)
        d_fake = discriminator(fake_imgs)
        loss_G = criterion(d_fake, real_labels)  

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if (i + 1) % 200 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Step [{i + 1}/{len(dataloader)}] | "
                  f"D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        sample_imgs = generator(z).view(-1, 1, 28, 28).cpu()
        grid = torch.cat([img for img in sample_imgs], dim=2)
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.show()
        torch.save(generator.state_dict(), "generator.pth")
print("Generator model saved.")

