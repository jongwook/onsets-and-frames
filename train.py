from datetime import datetime

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import *
from utils import *

ex = Experiment('transynth')


@ex.config
def config():
    logdir = 'runs/' + datetime.now().strftime('%y%m%d-%H%M%S')
    batch_size = 64
    sequence_length = 65536
    iterations = 100000


@ex.automain
def train(logdir, batch_size, sequence_length, iterations):
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    ex.observers.append(FileStorageObserver.create(logdir))

    dataset = Maestro(sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    loop = tqdm(range(1, iterations + 1))

    for i, batch in zip(loop, cycle(loader)):
        pass
