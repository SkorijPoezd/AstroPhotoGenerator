import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, ngf, n_ch):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8, affine=False),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4, affine=False),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Conv2d(ngf * 2, ngf * 2, 5, 1, 2),
            nn.BatchNorm2d(ngf * 2, affine=False),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.Conv2d(ngf, ngf, 7, 1, 3),
            nn.BatchNorm2d(ngf, affine=False),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, n_ch, 4, 2, 1, bias=False),
            nn.Conv2d(n_ch, n_ch, 7, 1, 3),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.cnt_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, n_ch, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_ch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.cnt_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, im):
        return self.main(im)
