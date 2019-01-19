"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from numpy import inf
from torch import nn


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, 48, (3, 3), padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(48, 48, (3, 3), padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(48, 96, (3, 3), padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            chunk_length = 1024

            # forward direction
            slices = range(0, sequence_length, chunk_length)
            for start in slices:
                end = start + chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class OnsetLoss(nn.Module):
    def forward(self, onset_pred, onset_label):
        return F.binary_cross_entropy(onset_pred, onset_label)


class FrameLoss(nn.Module):
    def forward(self, frame_pred, frame_label, ramp_label):
        weights = 5.0 / ramp_label
        weights[weights == inf] = 1.0
        return F.binary_cross_entropy(frame_pred, frame_label, weight=weights)


class VelocityLoss(nn.Module):
    def forward(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, hidden_features=768):
        super().__init__()

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, hidden_features),
            BiLSTM(hidden_features, hidden_features // 2),
            nn.Linear(hidden_features, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, hidden_features),
            nn.Linear(hidden_features, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            BiLSTM(output_features * 2, hidden_features // 2),
            nn.Linear(hidden_features, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, hidden_features),
            nn.Linear(hidden_features, output_features)
        )

        self.onset_loss = OnsetLoss()
        self.frame_loss = FrameLoss()
        self.velocity_loss = VelocityLoss()

    def forward(self, mel):
        onset_probs = self.onset_stack(mel)
        activation_probs = self.frame_stack(mel)
        combined_probs = torch.cat([onset_probs, activation_probs], dim=-1)
        frame_probs = self.combined_stack(combined_probs)
        velocity_probs = self.velocity_stack(mel)
        return onset_probs, frame_probs, velocity_probs
