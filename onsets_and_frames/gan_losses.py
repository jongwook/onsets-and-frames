import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.beta import Beta


class GANLoss(nn.Module):
    def __init__(self, discriminator, loss_function, real_label=1.0, fake_label=0.0, mixup=0):
        super().__init__()

        self.discriminator = discriminator
        self.loss_function = loss_function
        self.mixup = mixup
        if mixup > 0:
            self.real_beta = Beta(1 + mixup, mixup)
            self.fake_beta = Beta(mixup, 1 + mixup)

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    def forward(self, real=None, fake=None, skip_fake_loss=False):
        real_loss = fake_loss = 0

        example = real if torch.is_tensor(real) else real[0]
        batch_shape = [example.shape[0]]
        expanded_shape = batch_shape + [1] * (example.dim() - 1)

        if real is not None:
            label = self.real_label.expand(batch_shape)

            if self.mixup > 0:
                beta = self.real_beta.expand(batch_shape).sample().to(example.device)
                expanded_beta = beta.view(expanded_shape)
                real = expanded_beta * real + (1 - expanded_beta) * fake
                label = beta * self.real_label + (1 - beta) * self.fake_label

            real_scores = self.discriminator(real)
            label = label.view_as(real_scores)
            real_loss = self.loss_function(real_scores, label)

        if not skip_fake_loss and fake is not None:
            label = self.fake_label.expand(batch_shape)

            if self.mixup > 0:
                beta = self.fake_beta.expand(batch_shape).sample().to(example.device)
                expanded_beta = beta.view(expanded_shape)
                fake = expanded_beta * real + (1 - expanded_beta) * fake
                label = beta * self.real_label + (1 - beta) * self.fake_label

            fake_scores = self.discriminator(fake)
            label = label.view_as(fake_scores)
            fake_loss = self.loss_function(fake_scores, label)

        return real_loss, fake_loss


class VanillaGANLoss(GANLoss):
    def __init__(self, discriminator, real_label=1.0, fake_label=0.0, mixup=0):
        super().__init__(discriminator, F.binary_cross_entropy_with_logits, real_label, fake_label, mixup)


class LSGANLoss(GANLoss):
    def __init__(self, discriminator, real_label=1.0, fake_label=0.0, mixup=0):
        super().__init__(discriminator, F.mse_loss, real_label, fake_label, mixup)


class WGANGPLoss(GANLoss):
    def __init__(self, discriminator, real_label=1.0, fake_label=0.0, mixup=0, constant=1.0, lambda_gp=10.0):
        super().__init__(discriminator, self.wgan_loss, real_label, fake_label, mixup)
        self.constant = constant
        self.lambda_gp = lambda_gp

    @staticmethod
    def wgan_loss(prediction, label):
        return ((label - 0.5).sign() * prediction).mean()

    def forward(self, real=None, fake=None, skip_fake_loss=False):
        real_loss, fake_loss = super().forward(real, fake, skip_fake_loss)

        assert real is not None and fake is not None

        batch_size = real.shape[0]

        alpha = torch.rand(batch_size, 1, device=real.device)
        alpha = alpha.expand(batch_size, real.nelement() // real.shape[0])
        alpha = alpha.contiguous().view(*real.shape)
        interpolated = alpha * real + ((1 - alpha) * fake)
        interpolated.requires_grad_(True)

        interpolated_scores = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=interpolated_scores, inputs=interpolated,
                                        grad_outputs=torch.ones_like(interpolated_scores),
                                        create_graph=True, retain_graph=True, only_inputs=True)

        gradients = gradients[0].view(batch_size, -1)  # flatten the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - self.constant) ** 2).mean()

        return real_loss, fake_loss, self.lambda_gp * gradient_penalty
