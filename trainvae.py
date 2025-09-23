""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image

from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset
from torch.cuda.amp import autocast, GradScaler


parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

torch.manual_seed(123)
# if args.consistent_input_sizes:
#     torch.backends.cudnn.benchmark = True


scaler = GradScaler() if cuda else None

transform_train = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((RED_SIZE, RED_SIZE)),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
])

transform_test = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((RED_SIZE, RED_SIZE)),
    v2.ToTensor(),
])

dataset_train = RolloutObservationDataset('datasets/carracing',
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing',
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)


model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logsigma):
#     """ VAE loss function """
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum')

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     # KL divergence using distributions
#     posterior = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
#     prior = torch.distributions.Normal(0, 1)
#     kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum()
#     return recon_loss + kl_loss

def loss_function(recon_x, x, mu, logvar, beta=1.0, return_parts=False):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # Gaussian decoder likelihood
    logvar = torch.clamp(logvar, -20.0, 20.0)
    std = torch.exp(0.5 * logvar)
    posterior = torch.distributions.Normal(mu, std + 1e-8)
    prior = torch.distributions.Normal(0.0, 1.0)
    kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum()
    total = recon_loss + beta * kl_loss
    return (total, recon_loss, kl_loss) if return_parts else total

# ---- KL weight schedules ----
def linear_beta(epoch: int, warmup_epochs: int) -> float:
    # β ramps 0 → 1 over warmup_epochs, then stays at 1
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))

def cyclical_beta(epoch: int, cycle_len: int, floor: float = 0.0, ceil: float = 1.0) -> float:
    # β cycles from floor → ceil each cycle_len epochs
    if cycle_len <= 0:
        return ceil
    t = (epoch % cycle_len) / float(cycle_len)
    return floor + (ceil - floor) * t



def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    beta = linear_beta(epoch, warmup_epochs=50)  # warm-up for 50 epochs
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta=beta, return_parts=True)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr'] 
            bs = len(data)
            print(f"Train Epoch: {epoch} [{batch_idx*bs}/{len(train_loader.dataset)} "
                  f"({100.*batch_idx/len(train_loader):.0f}%)] "
                  f"loss/img={loss.item()/bs:.4f} recon/img={recon_loss.item()/bs:.4f} "
                  f"kl/img={kl_loss.item()/bs:.4f} beta={beta:.3f} lr={lr:.2e}")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
