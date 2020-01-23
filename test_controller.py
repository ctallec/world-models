""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch
import logging
from os import mkdir, unlink, listdir, getpid, kill


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

tmp_dir = join(args.logdir, 'tmp')
handler = logging.FileHandler(join(tmp_dir, 'mainlogger.out'))        
mainlogger = logging.getLogger('main_'+str(getpid()))


generator = RolloutGenerator(args.logdir, device, 1000, mainlogger)

try:
    with torch.no_grad():
        generator.rollout(None)
except Exception:
    mainlogger.error(f"Fatal error in main", exc_info=True)

