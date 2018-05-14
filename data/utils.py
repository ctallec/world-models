""" Some data loading utilities """
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import numpy as np

class RolloutSequenceDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    """ Encapsulate rollouts """
    def __init__(self, root, transform, train=True):
        self.transform = transform
        self.files = {}

        i = 0
        for sd in listdir(root):
            subdir = join(root, sd)
            if isdir(subdir):
                for ssd in listdir(subdir):
                    dfile = join(subdir, ssd)
                    self.files[i] = dfile
                    i += 1

        self.files = list(self.files)

        if train:
            self.files = self.files[:-500]
        else:
            self.files = self.files[-500:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wrapped = np.load(self.files[i])
        data = [wrapped[key].astype(np.float32) for key in
                ('actions', 'observations', 'rewards', 'terminals')]
        data[1] = self.transform(data[1])
        return data

# TODO : We should not concatenate all np arrays !

class RolloutObservationDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    """ Encapsulate individual images from rollouts """
    def __init__(self, root, extension, transform, buffersize):
        self.tranform = transform
        self.buffersize = buffersize

        subdirs = [join(root, sd)  for sd in listdir(root) if isdir(join(root, sd))]
        self.filenames = [join(sd, fn)
                          for sd in subdirs
                          for fn in listdir(sd) if fn.endswith(extension)]
        self.countfiles = 0
        self.countsamplesbuffer = 0
        self.observations = None

    def load_buffer(self):
        """ Load a different buffer (i.e. a different part of the training images) """
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
