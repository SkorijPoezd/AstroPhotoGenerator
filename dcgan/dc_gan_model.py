import torch.nn as nn
import torch


class DCGAN:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, device: str, z_dim: int,
                 generator_lr=1e-3, discriminator_lr=2e-4):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        self.z_dim = z_dim

        self.criterion = torch.nn.BCELoss()

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=generator_lr)
        self.D_optim = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr)

        self.cnt_parameters = {'G': sum(p.numel() for p in self.G.parameters() if p.requires_grad),
                               'D': sum(p.numel() for p in self.D.parameters() if p.requires_grad)
                               }

    def get_labels(self, batch_size: int):
        real_label = torch.full((batch_size, 1), 1.).to(self.device)
        fake_label = torch.full((batch_size, 1), 0.).to(self.device)
        return real_label, fake_label

    def train_gen_step(self, batch):
        self.G_optim.zero_grad()

        real_label, _ = self.get_labels(batch.shape[0])

        z = torch.randn(batch.shape[0], self.z_dim, 1, 1).to(self.device)

        fake_im = self.G(z)
        fake_im_prob = self.D(fake_im).view(-1, 1)

        G_loss = self.criterion(fake_im_prob, real_label)
        G_loss.backward()

        self.G_optim.step()

        return G_loss.item()

    def train_dis_step(self, batch):
        real_label, fake_label = self.get_labels(batch.shape[0])
        self.D_optim.zero_grad()

        # Train D on fake images
        z = torch.randn(batch.shape[0], self.z_dim, 1, 1).to(self.device)

        fake_im = self.G(z)
        fake_im_prob = self.D(fake_im).view(-1, 1)

        fake_D_loss = self.criterion(fake_im_prob, fake_label)

        # Train D on real images
        real_im_prob = self.D(batch.to(self.device)).view(-1, 1)
        real_D_loss = self.criterion(real_im_prob, real_label)

        D_loss = 0.5 * real_D_loss + 0.5 * fake_D_loss
        D_loss.backward()

        self.D_optim.step()

        return D_loss.cpu().item(), (real_D_loss.cpu().item(), fake_D_loss.cpu().item())

