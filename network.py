import torch
import torch.nn as nn

img_shape = [1, 64, 64]
structure_shape = [64, 64, 64]
nz = 400

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        # [bs, 1, 64, 64] -> [bs, nz]
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(512, nz)
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std, dtype = torch.float32, requires_grad = True).to(std.get_device())
        return mu + std * eps

    def forward(self, x):
        out = self.net(x)
        mu, logvar = torch.split(out, nz // 2, dim = 1)
        E = self.reparameterize(mu, logvar)
        return E, mu, logvar


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # [bs, nz // 2, 1, 1, 1] -> [bs, 1, 64, 64, 64]
        self.net = nn.Sequential(
            nn.ConvTranspose3d(nz // 2, 256, 4),
            nn.BatchNorm3d(256),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(32, 1, 4, 2, 1),
            # nn.Tanh()
        )

    def forward(self, x):
        # x: [bs, nz // 2]
        x = x.view(-1, nz // 2, 1, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [bs, 1, 64, 64, 64] -> [bs]
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(32, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(256, 256, 4),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)



def weight_init(net):
    for module in net.modules():
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def open_dropout(net):
    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.train()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    imgenc = ImageEncoder().to(device)
    x1 = torch.zeros([2, 1, 64, 64]).to(device)
    mu, logvar = imgenc(x1)
    print(mu.size(), logvar.size())

    gene = Generator().to(device)
    x2 = torch.zeros([2, nz // 2]).to(device)
    y2 = gene(x2)
    print(y2.size())

    disc = Discriminator().to(device)
    x3 = torch.zeros([2, 1, 64, 64, 64]).to(device)
    y3 = disc(x3)
    print(y3.size())
