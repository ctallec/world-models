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
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils import load_parameters
from utils import flatten_parameters, unflatten_parameters

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
        r_gen = RolloutGenerator(args.logdir, device, time_limit)

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
