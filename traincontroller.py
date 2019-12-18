"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, kill
from time import sleep
import signal
import torch
import pickle as pkl

import cma
from models import MDRNNCell, VAE, Controller

from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters, save_checkpoint
from utils.misc import flatten_parameters
from utils.history import History

import logging



################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index, logdir):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    
    # device = torch.device('cpu')
    # redirect streams
    tmp_dir = join(logdir, 'tmp', str(getpid()))
    sys.stdout = open(tmp_dir + '.out', 'a')
    sys.stderr = open(tmp_dir + '.err', 'a')


    handler = logging.FileHandler(tmp_dir + '.out')    
    logger = logging.getLogger('main_'+str(getpid()))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    try:
        with torch.no_grad():
            time_limit = 1000
            r_gen = RolloutGenerator(logdir, device, time_limit, logger)
            while e_queue.empty():
                if p_queue.empty():
                    sleep(.1)
                else:
                    s_id, params = p_queue.get()
                    r_queue.put((s_id, r_gen.rollout(params)))
    except Exception:
        logger.error(f"Fatal error in process {p_index}", exc_info=True)


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(param, rollouts, p_queue, r_queue):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """

    results = []

    for s_id in range(rollouts):
        p_queue.put((s_id, param))

    
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        results.append(r_queue.get()[1])

    return np.mean(results), np.std(results)


def init_random_models():
    cuda = torch.cuda.is_available()

    device = torch.device("cuda" if cuda else "cpu")

    vae = VAE(3, LSIZE).to(device)
    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)

    for (model, dirname) in [(vae, 'vae'), (mdrnn, 'mdrnn')]:
        
        best_filename = join(args.logdir, dirname, 'best.tar')
        if exists(best_filename):
            continue
        print(f"reinitilizing {dirname}")
        if not exists(join(args.logdir, dirname)):
            mkdir(join(args.logdir, dirname))

        filename = join(args.logdir, dirname, 'checkpoint.tar')
        checkpoint = {'epoch': 0, 'state_dict': model.state_dict()}
        if dirname in ['vae', 'mdrnn']:
            checkpoint['precision'] = np.inf
        else:
            checkpoint['reward'] = -np.inf
        save_checkpoint(checkpoint, True, filename, best_filename)
    
        




def main():
    ################################################################################
    # Initialization of the random models
    init_random_models()

    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = mp.Queue()
    r_queue = mp.Queue()
    e_queue = mp.Queue()

    p_list = []
    
    for p_index in range(num_workers):
        p_list.append(mp.Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, args.logdir)).start())

    
    ################################################################################
    #                           Launch CMA                                         #
    ################################################################################
    controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance

    # define current best and load parameters
    cur_best = None
    best_ctrl_file = join(ctrl_dir, 'best.tar')
    last_ctrl_file = join(ctrl_dir, 'last.tar')
    cma_file = join(ctrl_dir, 'cma.pkl')
    mainlogger.info("Attempting to load previous best...")
    epoch = 0
    if not args.noreload and exists(last_ctrl_file):
        state = torch.load(last_ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        epoch = state["epoch"] + 1
        controller.load_state_dict(state['state_dict'])
        mainlogger.info("Previous best was {}...".format(-cur_best))
        history.cut(epoch)

        with open(cma_file, 'rb') as f:
            es = pkl.load(f)

    else:
        es = cma.CMAEvolutionStrategy(flatten_parameters(controller.parameters()), 0.1,
                                      {'popsize': pop_size})

    log_step = 1
    while not es.stop() and epoch < args.max_epoch:
        if cur_best is not None and - cur_best > args.target_return:
            mainlogger.info("Already better than target, ointing...")
            break


        r_list = [0] * pop_size  # result list
        mainlogger.info("Ask CMA")
        solutions = es.ask()

        mainlogger.info("Put in the queue")
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        # retrieve results
        if args.display:
            pbar = tqdm(total=pop_size * n_samples)
        for t in range(pop_size * n_samples):
            mainlogger.info(f"{t}/{pop_size * n_samples}")
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if args.display:
                pbar.update(1)
        if args.display:
            pbar.close()
        mainlogger.info("Training completed. Now, CMA.")


        history.push("Return", epoch, r_list)

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            history.dump()
            _, best_params = min(enumerate(solutions), key=lambda idx_p: r_list[idx_p[0]])
            best, std_best = evaluate(best_params, 100, p_queue, r_queue)

            mainlogger.info("Current evaluation: {}".format(best))


            torch.save(
                {'epoch': epoch,
                 'reward': - best,
                 'state_dict': controller.state_dict()},
                last_ctrl_file)
            with open(cma_file, 'wb') as f:
                pkl.dump(es, f)

            if not cur_best or cur_best > best:
                cur_best = best
                mainlogger.info("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    best_ctrl_file)
            


        epoch += 1
        mainlogger.info("End of loop")

    for p in mp.active_children():
        kill(p.pid, signal.SIGKILL)

    print("end of killings")


    while len(mp.active_children()) == 0:#any(p.is_alive() for p in p_list if p is not None):
        sleep(1.)
    print("end of while loop")


    es.result_pretty()
    




if __name__ == '__main__':
    # parsing
    # torch.multiprocessing.set_start_method("spawn")
    mp = torch.multiprocessing.get_context('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where everything is stored.')
    parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain '
                        'return estimate.')
    parser.add_argument('--pop-size', type=int, help='Population size.')
    parser.add_argument('--target-return', type=float, help='Stops once the return '
                        'gets above target_return')
    parser.add_argument('--display', action='store_true', help="Use progress bars if "
                        "specified.")
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers.',
                        default=32)
    parser.add_argument('--noreload', action='store_true',
                help='Restart from scratch for the controller')
    parser.add_argument('--random-rnn', action='store_true',
                help='Do not load a trained RNN but use a random one')
    parser.add_argument('--random-vae', action='store_true',
                help='Do not load the trained RNN VAE but use a random one')
    parser.add_argument('--max_epoch', type=int, default=1000000)
    args = parser.parse_args()

    if args.random_rnn and args.random_vae:
        subdir = 'untrainedrnnvae'
    elif args.random_rnn and not args.random_vae:
        subdir = 'untrainedrnn'
    else:
        subdir = 'standard'


    # multiprocessing variables
    n_samples = args.n_samples
    pop_size = args.pop_size
    num_workers = min(args.max_workers, n_samples * pop_size)
    time_limit = 1000

    # mainlogger.info(f"CUDA:{torch.cuda.is_available()}, NGPU:{torch.cuda.device_count()}")
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
    history = History(join(ctrl_dir, 'history.pkl'))

    handler = logging.FileHandler(join(tmp_dir, 'mainlogger.out'))        
    mainlogger = logging.getLogger('main_'+str(getpid()))
    mainlogger.setLevel(logging.INFO)
    mainlogger.addHandler(handler)

    try:
        main()
    except Exception:
        mainlogger.error(f"Fatal error in main", exc_info=True)
    