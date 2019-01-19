import argparse
import os

import numpy as np
import torch
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from tqdm import tqdm

import dataset as dataset_module
from dataset import SAMPLE_RATE, HOP_LENGTH, MIN_MIDI
from utils import summary, save_pianoroll, extract_notes, notes_to_frames


def evaluate(model_file, dataset, dataset_group, sequence_length, save_piano_roll,
             onset_threshold, frame_threshold, device):
    # sequence_length = sequence_length if device == 'cpu' or sequence_length is not None else SAMPLE_RATE * 20
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    losses = []
    note_metrics = []
    note_with_offset_metrics = []
    note_with_velocity_metrics = []
    note_with_offset_and_velocity_metrics = []
    frame_metrics = []

    loop = tqdm(dataset)

    for data in loop:
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
        losses.append(dict(Lo=onset_loss.item(), Lf=frame_loss.item(), Lv=velocity_loss.item(), loss=loss.item()))

        onset_label = onset_label.squeeze(0)
        frame_label = frame_label.squeeze(0)
        velocity_label = velocity_label.squeeze(0)
        onset_pred = onset_pred.squeeze(0)
        frame_pred = frame_pred.squeeze(0)
        velocity_pred = velocity_pred.squeeze(0)

        if save_piano_roll is not None:
            os.makedirs(save_piano_roll, exist_ok=True)
            label_path = os.path.join(save_piano_roll, os.path.basename(data['path']) + '.label.png')
            save_pianoroll(label_path, onset_label, frame_label)
            pred_path = os.path.join(save_piano_roll, os.path.basename(data['path']) + '.pred.png')
            save_pianoroll(pred_path, onset_pred, frame_pred)

        ref_pitches, ref_intervals, ref_velocities = extract_notes(onset_label, frame_label, velocity_label)
        est_pitches, est_intervals, est_velocities = extract_notes(onset_pred, frame_pred, velocity_pred,
                                                                   onset_threshold, frame_threshold)

        ref_time, ref_freqs = notes_to_frames(ref_pitches, ref_intervals, frame_label.shape)
        est_time, est_freqs = notes_to_frames(est_pitches, est_intervals, frame_pred.shape)

        ref_intervals = np.array([[onset / HOP_LENGTH, offset / HOP_LENGTH] for onset, offset in ref_intervals])
        ref_pitches = np.array([midi_to_hz(MIN_MIDI + midi) for midi in ref_pitches])
        est_intervals = np.array([[onset / HOP_LENGTH, offset / HOP_LENGTH] for onset, offset in est_intervals])
        est_pitches = np.array([midi_to_hz(MIN_MIDI + midi) for midi in est_pitches])

        ref_time = np.array([time / HOP_LENGTH for time in ref_time])
        ref_freqs = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in ref_freqs]
        est_time = np.array([time / HOP_LENGTH for time in est_time])
        est_freqs = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in est_freqs]

        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)
        note_metrics.append(dict(Precision=p, Recall=r, F1=f, Overlap=o))

        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches)
        note_with_offset_metrics.append(dict(Precision=p, Recall=r, F1=f, Overlap=o))

        p, r, f, o = evaluate_notes_with_velocity(ref_intervals, ref_pitches, ref_velocities,
                                                  est_intervals, est_pitches, est_velocities,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        note_with_velocity_metrics.append(dict(Precision=p, Recall=r, F1=f, Overlap=o))

        p, r, f, o = evaluate_notes_with_velocity(ref_intervals, ref_pitches, ref_velocities,
                                                  est_intervals, est_pitches, est_velocities,
                                                  velocity_tolerance=0.1)
        note_with_offset_and_velocity_metrics.append(dict(Precision=p, Recall=r, F1=f, Overlap=o))

        metrics = evaluate_frames(ref_time, ref_freqs, est_time, est_freqs)
        frame_metrics.append(metrics)

    print('\nNote metrics:\n')
    for key in note_metrics[0].keys():
        metrics = [m[key] for m in note_metrics]
        print('%30s: %.3f ± %.3f' % (key, np.mean(metrics), np.std(metrics)))

    print('\nNote with offset metrics:\n')
    for key in note_with_offset_metrics[0].keys():
        metrics = [m[key] for m in note_with_offset_metrics]
        print('%30s: %.3f ± %.3f' % (key, np.mean(metrics), np.std(metrics)))

    print('\nNote with velocity metrics:\n')
    for key in note_with_velocity_metrics[0].keys():
        metrics = [m[key] for m in note_with_velocity_metrics]
        print('%30s: %.3f ± %.3f' % (key, np.mean(metrics), np.std(metrics)))

    print('\nNote with offset & velocity metrics:\n')
    for key in note_with_offset_and_velocity_metrics[0].keys():
        metrics = [m[key] for m in note_with_offset_and_velocity_metrics]
        print('%30s: %.3f ± %.3f' % (key, np.mean(metrics), np.std(metrics)))

    print('\nFrame metrics:\n')
    for key in frame_metrics[0].keys():
        metrics = [m[key] for m in frame_metrics]
        print('%30s: %.3f ± %.3f' % (key, np.mean(metrics), np.std(metrics)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-piano-roll', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        evaluate(**vars(parser.parse_args()))
