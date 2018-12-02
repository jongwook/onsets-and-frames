from datetime import datetime

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config, print_dependencies
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import *
from utils import *
from models.transcriber import OnsetsAndFrames

ex = Experiment('transynth')


@ex.config
def config():
    logdir = 'runs/' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 100000

    batch_size = 8
    sequence_length = SAMPLE_RATE * 20

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations,
          batch_size, sequence_length):
    print_dependencies(ex.current_run)
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    dataset = Maestro(sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    model = OnsetsAndFrames(229, 88).to(device)
    summary(model)

    loop = tqdm(range(1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        break
