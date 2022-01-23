import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # in-place operation
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def clip_weights(m, c=0.01):
    if hasattr(m, 'weight'):
        m.weight.data.clamp_(-c, c) # in-place operation
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.clamp_(-c, c) # in-place operation

def wasserstein_loss(real_samples, fake_samples, discriminator):
    scores_f = discriminator(fake_samples) # N x 1
    scores_r = discriminator(real_samples) # N x 1

    discriminator_loss = - (scores_r.mean() - scores_f.mean())
    return discriminator_loss
