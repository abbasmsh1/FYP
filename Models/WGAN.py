import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# WGAN Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, image_channels, hidden_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, hidden_dim * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

# WGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.net(x)

# Hyperparameters
# Define dataset transform
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Hyperparameters
noise_dim = 100
image_channels = 3
hidden_dim = 64
lr = 0.0002
betas = (0.5, 0.999)
n_critic = 5
num_epochs = 50
batch_size = 64
clip_value = 0.01

# Initialize CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate models and move to CUDA if available
generator = Generator(noise_dim, image_channels, hidden_dim).to(device)
discriminator = Discriminator(image_channels, hidden_dim).to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Training loop (you need to define your dataset loading)
for epoch in range(num_epochs):
    for batch_idx, real_images in enumerate(dataloader):
        # Train Discriminator
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            noise = torch.randn(batch_size, noise_dim, 1, 1,device=device)
            fake_images = generator(noise)
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images.detach())
            loss_D = -(torch.mean(real_validity) - torch.mean(fake_validity))
            loss_D.backward()
            optimizer_D.step()
            # Clip discriminator weights
            for param in discriminator.parameters():
                param.data.clamp_(-clip_value, clip_value)
        
        # Train Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, noise_dim, 1, 1,device = device)
        fake_images = generator(noise)
        fake_validity = discriminator(fake_images)
        loss_G = -torch.mean(fake_validity)
        loss_G.backward()
        optimizer_G.step()
