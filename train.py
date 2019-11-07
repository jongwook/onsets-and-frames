import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    gan_type = None  # otherwise 'wgan-gp', 'lsgan', or 'vanilla'
    gan_critic_iterations = 5 if gan_type == 'wgan-gp' else 1
    gan_real_label = 1.0
    gan_fake_label = 0.0
    gan_mixup = 0.0
    gan_gp_lambda = 10.0

    lambda_pix2pix = 100.0

    discriminator_optimizer = 'adam'
    discriminator_learning_rate = 0.0001

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          gan_type, gan_critic_iterations, gan_real_label, gan_fake_label, gan_mixup, gan_gp_lambda, lambda_pix2pix,
          discriminator_optimizer, discriminator_learning_rate,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    if gan_type is None:
        gan_loss = None
        discriminator = None
        discriminator_optimizer = None
        discriminator_scheduler = None
    else:
        discriminator = FullyConvolutionalDiscriminator().to(device)

        if gan_type == 'vanilla':
            gan_loss = VanillaGANLoss(discriminator, gan_real_label, gan_fake_label, gan_mixup)
        elif gan_type == 'lsgan':
            gan_loss = LSGANLoss(discriminator, gan_real_label, gan_fake_label, gan_mixup)
        elif gan_type == 'wgan-gp':
            gan_loss = WGANGPLoss(discriminator, gan_real_label, gan_fake_label, gan_mixup, lambda_gp=gan_gp_lambda)
        else:
            raise RuntimeError(f'Unsupported GAN type: {gan_type}')

        gan_loss.to(device)
        discriminator_optimizer_class = torch.optim.Adam if discriminator_optimizer == 'adam' else torch.optim.SGD
        discriminator_optimizer = discriminator_optimizer_class(discriminator.parameters(), discriminator_learning_rate)
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

        summary(discriminator)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    data_iterator = cycle(loader)
    for i in loop:
        batch = None
        transcriber_loss = None

        # train discriminator
        if gan_loss is not None:
            losses = dict()

            for j in range(gan_critic_iterations):
                batch = next(data_iterator)
                predictions, losses = model.run_on_batch(batch)
                transcriber_loss = sum(losses.values())

                real_predictions = torch.stack((batch['onset'], batch['frame']), dim=1)
                fake_predictions = torch.stack((predictions['onset'].detach(), predictions['frame'].detach()), dim=1)
                gan_losses = gan_loss(real_predictions, fake_predictions)
                loss = sum(gan_losses)

                losses['loss/discriminator'] = loss
                losses['loss/discriminator/real'] = gan_losses[0]
                losses['loss/discriminator/fake'] = gan_losses[1]

                if len(gan_losses) > 2:
                    losses['loss/discriminator/penalty'] = gan_losses[2]

                discriminator_optimizer.zero_grad()
                loss.backward()

                if clip_gradient_norm:
                    clip_grad_norm_(discriminator.parameters(), clip_gradient_norm)

                discriminator_optimizer.step()
                discriminator_scheduler.step()

            for key, value in losses.items():
                writer.add_scalar(key, value.item(), global_step=i)

        if batch is None:
            batch = next(data_iterator)
            predictions, losses = model.run_on_batch(batch)
            transcriber_loss = sum(losses.values())
        else:
            # reuse the predictions and losses from the discriminator step above
            losses = dict()  # skip the logs that are already written above

        losses['loss/generator/transcriber'] = transcriber_loss
        loss = transcriber_loss

        if gan_loss is not None:
            real_predictions = torch.stack((batch['onset'], batch['frame']), dim=1)
            fake_predictions = torch.stack((predictions['onset'], predictions['frame']), dim=1)

            gan_losses = gan_loss(fake_predictions, real_predictions, skip_fake_loss=True)
            penalized_gan_loss = gan_losses[0] + gan_losses[2] if len(gan_losses) > 2 else gan_losses[0]

            loss = transcriber_loss * lambda_pix2pix + penalized_gan_loss
            losses['loss/generator/discriminator'] = penalized_gan_loss
            losses['loss/generator/discriminator/gan'] = gan_losses[0]

            if len(gan_losses) > 2:
                losses['loss/generator/discriminator/penalty'] = gan_losses[2]

        losses['loss/generator'] = loss

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        optimizer.step()
        scheduler.step()

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

            if discriminator is not None:
                torch.save(discriminator.state_dict(), os.path.join(logdir, 'last-discriminator-state.pt'))
                torch.save(discriminator_optimizer.state_dict(), os.path.join(logdir, 'last-discriminator-optimizer-state.pt'))
