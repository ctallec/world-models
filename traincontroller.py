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
import torch

import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

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

    logger.info(f"Initialiazing cuda with {torch.cuda.device_count()} gpus")
    gpu = p_index % torch.cuda.device_count()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    logger.info("Initialiazing cuda Done")
    try:
        with torch.no_grad():
            time_limit = 1000
            r_gen = RolloutGenerator(logdir, device, time_limit, logger)
            logger.info("Starting the loop!")
            while e_queue.empty():
                if p_queue.empty():
                    sleep(.1)
                else:
                    logger.info("Queue is not empty")
                    s_id, params = p_queue.get()
                    logger.info("Got stuff from the queue, computing X")
                    X = r_gen.rollout(params)
                    logger.info("Putting X in the queue")
                    r_queue.put((s_id, X))
                    logger.info("X is in the queue")
    except Exception:
        logger.error(f"Fatal error in process {p_index}", exc_info=True)


# def mpify(target, num_workers, nruns, parameters, args_target, logger, margin=2, display=False):
#     p_queue = mp.Queue()
#     r_queue = mp.Queue()
#     e_queue = mp.Queue()

#     # torch.multiprocessing.spawn(fn=slave_routine, args=(p_queue, r_queue, e_queue), 
#     #     nprocs=num_workers, join=False)
#     process_list = []
#     for p_index in range(num_workers):
#         p = mp.Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, *args_target))
#         process_list.append(p)
#         p.start()
#         # mp.Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, *args_target)).start()



#     results = [[] for _ in parameters]
#     for s_id, s in enumerate(parameters):
#         for _ in range(n_samples + margin):
#             p_queue.put((s_id, s))
    

#     if display:
#         pbar = tqdm(total=len(parameters) * nruns)

#     count = 0
#     while any(len(r_list) < nruns for r_list in results):
#         if p_queue.empty():
#             for idx, p in sorted(enumerate(parameters), key= lambda idx_p: len(results[idx_p[0]])):
#                 if len(results[idx]) >= nruns:
#                     break
#                 p_queue.put((s_id, s))

#         # for t in range(pop_size * n_samples):
#         # logger.info(f"{count}/{len(parameters) * nruns}")
#         while r_queue.empty():
#             sleep(.1)
#         r_s_id, r = r_queue.get()

#         if len(results[r_s_id]) < nruns:
#             results[r_s_id].append(r)
#             if args.display:
#                 pbar.update(1)
#             count += 1
        
#     e_queue.put('EOP')
#     sleep(1.)
#     for p in process_list:
#         p.terminate()            

#     if args.display:
#         pbar.close()

#     return results

        
    



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
    mainlogger.info("Evaluating...")
    # index_min = np.argmin(results)
    # best_guess = solutions[index_min]
    # results = mpify(target=slave_routine, num_workers=args.max_workers, nruns=rollouts, 
    #     parameters=[param], args_target=(args.logdir,), logger=mainlogger, 
    #     display=args.display)
    
    # assert len(results) == 1
    # results = np.array(results[0])

    results = []

    for s_id in range(rollouts):
        p_queue.put((s_id, param))

    
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        results.append(r_queue.get()[1])

    return np.mean(results), np.std(results)


def main():
    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = mp.Queue()
    r_queue = mp.Queue()
    e_queue = mp.Queue()

    # torch.multiprocessing.spawn(fn=slave_routine, args=(p_queue, r_queue, e_queue), 
    #     nprocs=num_workers, join=False)
    for p_index in range(num_workers):
        mp.Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, args.logdir)).start()


    ################################################################################
    #                           Launch CMA                                         #
    ################################################################################
    controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance

    # define current best and load parameters
    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    mainlogger.info("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        mainlogger.info("Previous best was {}...".format(-cur_best))

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': pop_size})

    epoch = 0
    log_step = 1
    mainlogger.info("Beginning of training")
    while not es.stop():
        if cur_best is not None and - cur_best > args.target_return:
            mainlogger.info("Already better than target, breaking...")
            break


        r_list = [0] * pop_size  # result list
        mainlogger.info("Ask CMA")
        solutions = es.ask()


        # results = mpify(target=slave_routine, num_workers=args.max_workers, nruns=n_samples, 
        #     parameters=solutions, args_target=(args.logdir,), logger=mainlogger, 
        #     display=args.display)
        # push parameters to queue
        mainlogger.info("Put in the queue")
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples + 2):
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

        # r_list = [np.mean(r) for r in results]
        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            
            _, best_params = min(enumerate(solutions), key=lambda idx_p: r_list[idx_p[0]])
            best, std_best = evaluate(best_params, 100)

            mainlogger.info("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                mainlogger.info("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    join(ctrl_dir, 'best.tar'))
            


        epoch += 1
        mainlogger.info("End of loop")
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
    args = parser.parse_args()

    args.logdir = "/private/home/leonardb/code/world-models/exp_dir"
    # Max number of workers. M

    


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

    handler = logging.FileHandler(join(tmp_dir, 'mainlogger.out'))        
    mainlogger = logging.getLogger('main_'+str(getpid()))
    mainlogger.setLevel(logging.INFO)
    mainlogger.addHandler(handler)

    try:
        main()
    except Exception:
        mainlogger.error(f"Fatal error in main", exc_info=True)
    