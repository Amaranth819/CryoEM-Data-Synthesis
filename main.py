import torch
import torchvision
import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import *
# from net import *
from network import *

def total_param_num(net):
    num = 0
    for param in net.parameters():
        num += param.numel()
    return num

def train(config):
    # Configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Network
    img_encoder = ImageEncoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if os.path.exists(config.model_path + config.model_name):
        ckpt = torch.load(config.model_path + config.model_name)

        img_encoder.load_state_dict(ckpt['imgenc'])
        generator.load_state_dict(ckpt['gene'])
        discriminator.load_state_dict(ckpt['disc'])

        print('Load the pretrained model from %s successfully!' % (config.model_path + config.model_name))
    else:
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        weight_init(img_encoder)
        weight_init(generator)
        weight_init(discriminator)

        print('First time training!')

    img_encoder.train()
    generator.train()
    discriminator.train()

    print(total_param_num(img_encoder))
    print(total_param_num(generator))
    print(total_param_num(discriminator))
    return
        
    # Dataset
    structure_paths = sorted(search_by_re(config.root_dir, config.structure_re))
    proj_paths = sorted(search_by_re(config.root_dir, config.proj_re))
    print(len(structure_paths), len(proj_paths))
    train_pairs = StructureProjPairs(structure_paths, proj_paths)
    train_loader = DataLoader(train_pairs, batch_size = config.batch_size, shuffle = True, drop_last = True)

    # Optimizer
    optimizer_imgenc = torch.optim.Adam(img_encoder.parameters(), lr = 1e-4, betas = [0.5, 0.999])
    optimizer_gene = torch.optim.Adam(generator.parameters(), lr = 1e-3, betas = [0.5, 0.999])
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr = 1e-3, betas = [0.5, 0.999])

    scheduler_imgenc = torch.optim.lr_scheduler.StepLR(optimizer_imgenc, 20, 0.1)
    scheduler_gene = torch.optim.lr_scheduler.StepLR(optimizer_gene, 20, 0.1)
    scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, 20, 0.1)

    # Loss functions
    recon_loss = lambda input_, target_: torch.mean(torch.sum((input_ - target_)**2, dim = [1, 2, 3, 4]).sqrt_())
    bce_loss = torch.nn.BCELoss()
    # kld_loss = torch.nn.KLDivLoss()
    kld_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp_())

    # Summary writer
    summary = SummaryWriter(config.summary_path)

    # Paths
    if not os.path.exists(config.test_path):
        os.makedirs(config.test_path)
    if not os.path.exists(config.fake_path):
        os.makedirs(config.fake_path)

    # Start training
    total_iter = 1
    for e in range(config.epoch):
        for idx, (proj, structure) in enumerate(train_loader):
            # proj, structure = proj.to(device), structure.to(device)
            proj = proj.to(device)
            proj.requires_grad_(True)
            if np.random.rand() < 0.1:
                structure = (torch.rand(structure.size()) + structure).clone().detach()
            structure = structure.to(device)
            structure.requires_grad_(True)

            bs = structure.size(0)
            z = torch.randn([bs, nz // 2], dtype = torch.float32, requires_grad = True).to(device)
            if config.soft_and_noisy:
                real_labels = torch.rand([bs], dtype = torch.float32) * 0.5 + 0.7 if np.random.rand() > 0.05 else torch.rand([bs], dtype = torch.float32) * 0.3
                fake_labels = torch.rand([bs], dtype = torch.float32) * 0.3 # 0 - 0.3
            else:
                real_labels = torch.ones([bs], dtype = torch.float32) if np.random.rand() > 0.05 else torch.zeros([bs], dtype = torch.float32)
                fake_labels = torch.zeros([bs], dtype = torch.float32) # 0

            real_labels = real_labels.to(device)
            fake_labels = fake_labels.to(device)

            # Train the discriminator
            dloss = []
            for _ in range(3):
                optimizer_disc.zero_grad()
                real_pred = discriminator(structure)
                Gz = generator(z)
                fake_d_pred = discriminator(Gz.detach())
                dloss_i = bce_loss(real_pred, real_labels) + bce_loss(fake_d_pred, fake_labels)
                dloss.append(dloss_i.item())
                dloss_i.backward(retain_graph = True)
                optimizer_disc.step()

            # Train the image encoder
            optimizer_imgenc.zero_grad()
            Ey, mu, logvar = img_encoder(proj)
            GEy = generator(Ey)
            eloss1 = kld_loss(mu, logvar)
            eloss2 = recon_loss(GEy, structure)
            eloss = eloss1 + eloss2
            eloss.backward(retain_graph = True)
            optimizer_imgenc.step()

            # Train the generator
            optimizer_gene.zero_grad()
            fake_pred = discriminator(Gz)
            gloss1 = bce_loss(fake_pred, fake_labels)
            gloss2 = recon_loss(GEy, structure)
            gloss = gloss1 + gloss2 
            gloss.backward()
            optimizer_gene.step()

            print('[Epoch %d|Batch %d] dloss = %.5f, eloss = (%.5f, %.5f), gloss = (%.5f, %.5f)'
                    % (e, idx, np.mean(dloss), eloss1.item(), eloss2.item(), gloss1.item(), gloss2.item()))

            summary.add_scalar('Train/Dloss', np.mean(dloss), total_iter)
            summary.add_scalar('Train/ELoss1', eloss1.item(), total_iter)
            summary.add_scalar('Train/ELoss2', eloss2.item(), total_iter)
            summary.add_scalar('Train/GLoss1', gloss1.item(), total_iter)
            summary.add_scalar('Train/GLoss2', gloss2.item(), total_iter)
            total_iter += 1

        if e % 2 == 1:
            generator.eval()
            # open_dropout(generator)
            with torch.no_grad():
                noise = torch.randn([4, nz // 2], dtype = torch.float32).to(device)
                pred_structures = generator(noise).squeeze().detach().cpu().numpy()

            for i in range(pred_structures.shape[0]):
                save_mrc(pred_structures[i], config.test_path + 'e%d_b%d.mrc' % (e, i))

            generator.train()

        if e % 2 == 1:
            states = {
                'imgenc' : img_encoder.state_dict(),
                'gene' : generator.state_dict(),
                'disc' : discriminator.state_dict()
            }
            # torch.save(states, config.model_path + config.model_name)
            torch.save(states, config.model_path + 'model_e%d.pkl' % e)

        scheduler_imgenc.step()
        scheduler_gene.step()
        scheduler_disc.step()

    summary.close()


def fake(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    generator = Generator().to(device)
    model = torch.load(config.model_path + config.model_name)
    generator.load_state_dict(model['gene'])
    generator.eval()
    # open_dropout(generator)

    if not os.path.exists(config.fake_path):
        os.makedirs(config.fake_path)

    with torch.no_grad():
        # for i in range(16):
        #     z = torch.randn([1, nz //2], dtype = torch.float32).to(device)
        #     Gz = generator(z).squeeze().detach().cpu().numpy()
        #     save_mrc(Gz, config.fake_path + 'fake_%d.mrc' % i)

        count = 1
        for q in range(200):
            z = torch.randn([16, nz // 2], dtype = torch.float32).to(device)
            Gz = generator(z).squeeze().detach().cpu().numpy()
            for b in range(16):
                root = config.fake_path + '%d/' % count
                if not os.path.exists(root):
                    os.makedirs(root)
                    save_mrc(Gz[b], root + '%d.mrc' % count)
                count += 1

            print('Finish writing %d batches!' % q)
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--model_path', type = str, default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--root_dir', type = str, default = '../../dataset/testSet/')
    parser.add_argument('--structure_re', type = str, default = '*/*_64_norm.mrc')
    parser.add_argument('--proj_re', type = str, default = '*/*_64_norm_projs.mrcs')
    parser.add_argument('--summary_path', type = str, default = './summary/')
    parser.add_argument('--test_path', type = str, default = './test/')
    parser.add_argument('--fake_path', type = str, default = './fake/')
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--soft_and_noisy', type = bool, default = True)

    config = parser.parse_args()
    if config.is_train:
        train(config)
    else:
        fake(config)
