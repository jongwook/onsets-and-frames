import os
import argparse
import torch

import dataset as dataset_module
from utils import summary, save_pianoroll


def evaluate(model_file, dataset, save_piano_roll):
    dataset_class = getattr(dataset_module, dataset)
    dataset = dataset_class(sequence_length=16000 * 20)

    model = torch.load(model_file).eval()
    summary(model)

    for data in dataset:
        audio_label = data['audio'].unsqueeze(0)
        onset_label = data['onsets'].unsqueeze(0)
        frame_label = data['frames'].unsqueeze(0)
        velocity_label = data['velocities'].unsqueeze(0)
        ramp_label = data['ramps'].unsqueeze(0)
        mel_label = dataset.mel(audio_label[:, :-1]).transpose(-1, -2)

        onset_pred, frame_pred, velocity_pred = model(mel_label)
        onset_loss = model.onset_loss(onset_pred, onset_label)
        frame_loss = model.frame_loss(frame_pred, frame_label, ramp_label)
        velocity_loss = model.velocity_loss(velocity_pred, velocity_label, onset_label)
        loss = onset_loss + frame_loss + velocity_loss
        print(dict(Lo=onset_loss.item(), Lf=frame_loss.item(), Lv=velocity_loss.item(), loss=loss.item()))

        if save_piano_roll is not None:
            os.makedirs(save_piano_roll, exist_ok=True)
            label_path = os.path.join(save_piano_roll, os.path.basename(data['path']) + '.label.png')
            save_pianoroll(label_path, onset_label[0], frame_label[0])
            pred_path = os.path.join(save_piano_roll, os.path.basename(data['path']) + '.pred.png')
            save_pianoroll(pred_path, onset_pred[0], frame_pred[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAPS')
    parser.add_argument('--save-piano-roll', default=None)

    with torch.no_grad():
        evaluate(**vars(parser.parse_args()))
