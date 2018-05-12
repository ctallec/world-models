import argparse
import numpy as np
import os
from os.path import isdir, join

from tqdm import tqdm


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


from models.vae import Encoder, Decoder
from itertools import chain

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(123)

device = torch.device("cuda" if args.cuda else "cpu")


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# TODO : We should not concatenate all np arrays !

class RolloutObservationDataset(torch.utils.data.Dataset):
    def __init__(self, root, extension, transform, buffersize):
        self.tranform = transform
        self.buffersize = buffersize

        subdirs = [join(root, sd)  for sd in os.listdir(root) if isdir(join(root,sd))]
        self.filenames = [join(sd, fn) for sd in subdirs for fn in os.listdir(sd) if fn.endswith(extension)]
        self.countfiles = 0
        self.countsamplesbuffer = 0
        
    def load_buffer(self):
        filebuffer = self.filenames[self.countfiles:self.countfiles+self.buffersize]
        obslist = []
        pbar = tqdm(total=len(filebuffer),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")
        for fn in filebuffer:
            with np.load(fn) as data:
                obslist.append(data['observations'])
            pbar.update(1)
        self.observations = np.concatenate(obslist)
        pbar.close()
    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, i):
        self.countsamplesbuffer += 1
        return self.tranform(self.observations[i])

        
def observationloader(filename):
    with np.load(filename) as data:
        observation = data['observations']
    idx = np.random.randint(observation.shape[0])
    return observation[idx]
        


# dataset_train = datasets.DatasetFolder('/data/titanic_4/datasets/carracing/rollouts/',
#                                        observationloader, 'npz', transform = transform_train)

# dataset_test = datasets.DatasetFolder('/data/titanic_4/datasets/carracing/rollouts/',
#                                       observationloader, 'npz', transform = transform_test)
# train_loader = torch.utils.data.DataLoader(dataset_train,
#     batch_size=args.batch_size, shuffle=True, num_workers=2)
# test_loader = torch.utils.data.DataLoader(dataset_test,
#     batch_size=args.batch_size, shuffle=True, num_workers=2)

dataset_train = RolloutObservationDataset('/data/titanic_4/datasets/carracing/rollouts/',
                                          'npz', transform_train, 200)
dataset_test = RolloutObservationDataset('/data/titanic_4/datasets/carracing/rollouts/',
                                          'npz', transform_test, 10)
train_loader = torch.utils.data.DataLoader(dataset_train,
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset_test,
    batch_size=args.batch_size, shuffle=True, num_workers=2)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(3, 32)
        self.decoder = Decoder(3, 32)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
    

        
model = VAE().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    model.train()
    dataset_train.load_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    dataset_test.load_buffer()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 32).to(device)
        sample = model.decoder(sample).cpu()
        save_image(sample.view(64, 3, 64, 64),
                   'results/sample_' + str(epoch) + '.png')
