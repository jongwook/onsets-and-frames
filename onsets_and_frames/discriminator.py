import torch
from torch import nn

from .transcriber import ConvStack
from .lstm import BiLSTM


class RecurrentDiscriminator(nn.Module):
    """
        A conditional GAN discriminator that has similar CNN-LSTM architecture as in Onsets & Frames
    """
    def __init__(self, mel_features, roll_features, hidden_features=192):
        super().__init__()
        self.mel_cnn = ConvStack(mel_features, hidden_features)
        self.mel_rnn = nn.GRU(hidden_features, hidden_features // 2, batch_first=True, bidirectional=True)
        self.onset_cnn = ConvStack(roll_features, hidden_features)
        self.onset_rnn = nn.GRU(hidden_features, hidden_features // 2, batch_first=True, bidirectional=True)
        self.frame_cnn = ConvStack(roll_features, hidden_features)
        self.frame_rnn = nn.GRU(hidden_features, hidden_features // 2, batch_first=True, bidirectional=True)
        self.combiner = nn.Linear(hidden_features * 3, hidden_features)
        self.output = nn.Linear(hidden_features, 1)

    def forward(self, inputs):
        """

        Parameters
        ----------
        mel: torch.Tensor, shape=[batch, num_frames, mel_features]
        roll: torch.Tensor, shape=[batch, num_frames, roll_features]

        Returns
        -------
        torch.Tensor, shape=[batch, 1]

            contains logit score of this discriminator.
        """
        mel, onsets, frames = inputs
        mel = self.mel_rnn(self.mel_cnn(mel))[1]
        onsets = self.onset_rnn(self.onset_cnn(onsets))[1]
        frames = self.frame_rnn(self.frame_cnn(frames))[1]

        # hidden connections doesn't reflect batch_first
        mel = mel.transpose(0, 1)
        onsets = onsets.transpose(0, 1)
        frames = frames.transpose(0, 1)

        mel = mel.reshape(mel.shape[0], -1)
        onsets = onsets.reshape(onsets.shape[0], -1)
        frames = frames.reshape(frames.shape[0], -1)

        stack = torch.cat([mel, onsets, frames], dim=-1)
        combined = self.combiner(stack).relu()
        output = self.output(combined)

        return output


class FullyConvolutionalDiscriminator(nn.Module):
    """
        A GAN discriminator similar to the PatchGAN architecture.
    """
    def __init__(self):
        super().__init__()

        kernel_size = (3, 3)
        stride = 2
        padding = 1
        dropout = 0.5

        self.layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size, stride, padding),
            nn.Dropout(dropout), nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size, stride, padding),
            nn.Dropout(dropout), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size, stride, padding),
            nn.Dropout(dropout), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size, stride, padding),
            nn.Dropout(dropout), nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, (5, 5), padding=2)
        )

    def forward(self, input):
        return self.layers(input).mean(dim=(-2, -1))
