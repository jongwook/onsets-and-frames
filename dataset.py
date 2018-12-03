import json
import os
from abc import abstractmethod

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from midi import parse_midi

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 8 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)
        self.sequence_length = sequence_length
        self.device = device

        self.data = []
        print('Loading %d group%s of %s at %s' % (len(groups), 's'[:len(groups)-1], self.__class__.__name__, path))
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = {}

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = np.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['ramps'] = data['ramps'][:, step_begin:step_end].to(self.device)
            result['velocities'] = data['velocities'][:, step_begin:step_end].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['ramps'] = data['ramps'].to(self.device)
            result['velocities'] = data['velocities'].to(self.device)

        result['onsets'] = result['ramps'] == 1
        result['frames'] = result['ramps'] > 0

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files in a group"""
        raise NotImplementedError

    @abstractmethod
    def load(self, *args):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            ramp: torch.ByteTensor, shape = [midi_bins, num_steps]
                a matrix that contains the number of frames after the corresponding onset

            velocity: torch.ByteTensor, shape = [midi_bins, num_steps]
                a matrix that contains MIDI velocity values at the frame locations
        """
        raise NotImplementedError


class Maestro(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
        return sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                        os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

    def load(self, audio_filename, midi_filename):
        saved_data_path = audio_filename.replace('.flac', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_filename, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        ramp_template = torch.ByteTensor(HOPS_IN_ONSET + 254)
        ramp_template[:HOPS_IN_ONSET] = 1
        ramp_template[-254:] = torch.arange(2, 256, dtype=torch.uint8)

        ramps = torch.zeros(n_keys, n_steps, dtype=torch.uint8)
        velocities = torch.zeros(n_keys, n_steps, dtype=torch.uint8)

        tsv_path = midi_filename.replace('.midi', '.tsv').replace('.mid', '.tsv')
        if os.path.exists(tsv_path):
            midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        else:
            midi = parse_midi(midi_filename)

        for onset, offset, note, velocity in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            ramp_right = min(frame_right, left + len(ramp_template))

            f = int(note) - MIN_MIDI
            ramps[f, left:ramp_right] = ramp_template[:ramp_right - left]
            ramps[f, ramp_right:frame_right] = 255
            velocities[f, left:frame_right] = velocity

        data = dict(audio=audio, ramps=ramps, velocities=velocities)
        torch.save(data, saved_data_path)
        return data


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    # TODO: implement the other methods


if __name__ == '__main__':
    dataset = Maestro(sequence_length=65536)
    print(dataset[0])
