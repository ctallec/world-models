""" Recurrent model training """
import argparse
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils import save_checkpoint

from data.utils import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants
LSIZE = 32
BSIZE = 8
SEQ_LEN = 32
SIZE = 96
RED_SIZE = 64
log_step = 10
epochs = 30

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, 3, 256, 5)
mdrnn.to(device)
optimizer = torch.optim.Adam(mdrnn.parameters())

if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])


# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=200),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=200),
    batch_size=BSIZE, num_workers=8)

def train(epoch): # pylint: disable=too-many-locals
    """ One epoch of training """
    mdrnn.train()
    train_loader.dataset.load_next_buffer()

    cum_loss = 0
    pbar = tqdm(total=len(train_loader.dataset), desc="Epoch {}".format(epoch))

    for i, data in enumerate(train_loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        with torch.no_grad():
            obs, next_obs = [
                f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

        latent_obs, action,\
            reward, terminal,\
            latent_next_obs = [arr.transpose(1, 0)
                               for arr in [latent_obs, action,
                                           reward, terminal,
                                           latent_next_obs]]
        mus, sigmas, pi, rs, ds = mdrnn(action, latent_obs)
        loss = gmm_loss(latent_next_obs, mus, sigmas, pi)
        loss += f.binary_cross_entropy_with_logits(ds, terminal)
        loss += f.mse_loss(rs, reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()

        if i % log_step == log_step - 1:
            pbar.set_postfix_str("loss={loss:10.6f} avg_loss={avg_loss:10.6f}".format(
                loss=loss.item(), avg_loss=cum_loss / (i + 1)))
            pbar.update(log_step * BSIZE)
    pbar.close()

def test(epoch): # pylint: disable=too-many-locals
    """ One epoch of training """
    mdrnn.eval()
    test_loader.dataset.load_next_buffer()

    cum_loss = 0
    pbar = tqdm(total=len(test_loader.dataset), desc="Test, epoch {}".format(epoch))

    for i, data in enumerate(train_loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        with torch.no_grad():
            obs, next_obs = [
                f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear')
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

            latent_obs, action,\
                reward, terminal,\
                latent_next_obs = [arr.transpose(1, 0)
                                   for arr in [latent_obs, action,
                                               reward, terminal,
                                               latent_next_obs]]
            mus, sigmas, pi, rs, ds = mdrnn(action, latent_obs)
            loss = gmm_loss(latent_next_obs, mus, sigmas, pi)
            loss += f.binary_cross_entropy_with_logits(ds, terminal)
            loss += f.mse_loss(rs, reward)

            cum_loss += loss.item()

            pbar.set_postfix_str("loss={loss:10.6f} avg_loss={avg_loss:10.6f}".format(
                loss=loss.item(), avg_loss=cum_loss / (i + 1)))
            pbar.update(log_step * BSIZE)
        pbar.close()
        return cum_loss / (i + 1)

for e in range(epochs):
    cur_best = None
    train(e)
    test_loss = test(e)

    is_best = not cur_best or test_loss < cur_best
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)
