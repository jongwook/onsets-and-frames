from datetime import datetime

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config, print_dependencies
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import *
from utils import *
from models import OnsetsAndFrames

ex = Experiment('transynth')


@ex.config
def config():
    logdir = 'runs/' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 100000
    checkpoint_interval = 1000

    batch_size = 8
    sequence_length = 32768

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, checkpoint_interval,
          batch_size, sequence_length):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    dataset = Maestro(sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    model = OnsetsAndFrames(229, 88).to(device)
    summary(model)

    optimizer = torch.optim.Adam(model.parameters())

    loop = tqdm(range(1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        audio_label = batch['audio']
        onset_label = batch['onsets']
        frame_label = batch['frames']
        velocity_label = batch['velocities']
        ramp_label = batch['ramps']
        mel_label = dataset.mel(audio_label[:, :-1]).transpose(-1, -2)

        onset_pred, frame_pred, velocity_pred = model(mel_label)
        onset_loss = model.onset_loss(onset_pred, onset_label)
        frame_loss = model.frame_loss(frame_pred, frame_label, ramp_label)
        velocity_loss = model.velocity_loss(velocity_pred, velocity_label, onset_label)
        loss = onset_loss + frame_loss + velocity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(dict(Lo=onset_loss.item(), Lf=frame_loss.item(), Lv=velocity_loss.item()))

        writer.add_scalar('loss', loss.item(), global_step=i)
        writer.add_scalar('loss/onset', onset_loss.item(), global_step=i)
        writer.add_scalar('loss/frame', frame_loss.item(), global_step=i)
        writer.add_scalar('loss/velocity', velocity_loss.item(), global_step=i)

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, 'model-%d.pt' % i))
            torch.save(optimizer, os.path.join(logdir, 'model-%d.optimizer.pt' % i))
