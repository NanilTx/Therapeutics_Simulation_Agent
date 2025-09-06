from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class VAESettings:
    input_dim: int
    latent_dim: int = 16

class Encoder(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 128)
        self.mu = nn.Linear(128, d_latent)
        self.logvar = nn.Linear(128, d_latent)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, d_latent, d_out):
        super().__init__()
        self.fc = nn.Linear(d_latent, 128)
        self.out = nn.Linear(128, d_out)
    def forward(self, z):
        h = F.relu(self.fc(z))
        return self.out(h)

class SimpleVAE(nn.Module):
    def __init__(self, settings: VAESettings):
        super().__init__()
        self.enc = Encoder(settings.input_dim, settings.latent_dim)
        self.dec = Decoder(settings.latent_dim, settings.input_dim)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        rec = F.mse_loss(recon, x, reduction='none').sum(dim=1)
        loss = (kl + rec).mean()
        return recon, z, loss
