"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import torch.nn as nn
import cma
import gym
from models import VAE, MDRNNCell
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
args = parser.parse_args()

# Some constants
# TODO: put all constants in a config file
LSIZE = 32
RSIZE = 256
ASIZE = 3

# multiprocessing variables
n_samples = 4
pop_size = 32
num_workers = min(32, n_samples * pop_size)
time_limit = 1000

# create tmp dir if non existent and clean it if existent
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# create ctrl dir if non exitent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

################################################################################
#                           Sub functions                                      #
################################################################################

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)

def flatten_parameters(params):
    """ Flattening parameters """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ unflatten parameters """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """
    Encapsulate everything that is needed to
    generate rollouts.
    """
    def __init__(self, mdir, device):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v0')
        self.device = device

    def get_action_and_transition(self, obs, hidden):
        """ Gets action and transition """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params):
        """ One rollout """
        # copy params into the controller
        load_parameters(params, self.controller)

        obs = self.env.reset()
        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)
            cumulative += reward
            if done or i > time_limit:
                return - cumulative
            i += 1

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index):
    """ Thread routine.

    :args p_queue: queue containing couples (p_id, s_id, parameters) to evaluate
    :args r_queue: where to place results (p_id, s_id, results)
    :args e_queue: as soon as not empty, terminate
    """
    # init routine
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, device)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))

################################################################################
#                Define queues and start workers                               #
################################################################################
p_queue = Queue()
r_queue = Queue()
e_queue = Queue()

for p_index in range(num_workers):
    Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()

################################################################################
#                           Launch CMA                                         #
################################################################################
controller = Controller(LSIZE, RSIZE, ASIZE) # dummy instance
parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                              {'popsize': pop_size})

# define current best
cur_best = None
ctrl_file = join(ctrl_dir, 'best.tar')
if exists(ctrl_file):
    cur_best = - torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})['reward']
    print("Previous best was {}...".format(-cur_best))

epoch = 0
while not es.stop():
    r_list = [0] * pop_size # result list
    solutions = es.ask()

    # push parameters to queue
    for s_id, s in enumerate(solutions):
        for _ in range(n_samples):
            p_queue.put((s_id, s))

    # retrieve results
    pbar = tqdm(total=pop_size * n_samples)
    for _ in range(pop_size * n_samples):
        while r_queue.empty():
            sleep(.1)
        r_s_id, r = r_queue.get()
        r_list[r_s_id] += r / n_samples
        pbar.update(1)
    pbar.close()

    es.tell(solutions, r_list)
    es.disp()

    # save parameters
    index_best = np.argmin(r_list)
    best = r_list[index_best]
    if not cur_best or cur_best > best:
        cur_best = best
        print("Saving new best with value {}...".format(-cur_best))
        load_parameters(solutions[index_best], controller)
        torch.save(
            {'epoch': epoch,
             'reward': - cur_best,
             'state_dict': controller.state_dict()},
            join(ctrl_dir, 'best.tar'))
    epoch += 1

es.result_pretty()
e_queue.put('EOP')
