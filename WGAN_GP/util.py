import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02) # in-place operation
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def wasserstein_loss_gp(real_samples, fake_samples, discriminator, device, lambda_gp):
    '''
        compute the gradient of the critic scores w.r.t. the average input
        since we will have to compute the derivative of the gradient_norm values w.r.t. the discriminator parameters,
        we need to build a computation graph connecting the discriminator parameters to the gradient values
    '''
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).to(device)
    average = epsilon * real_samples + (1 - epsilon) * fake_samples

    output = discriminator(average)
    grad_average = torch.autograd.grad(output, average, torch.ones_like(output), create_graph=True )[0]

    grad_average = grad_average.reshape(BATCH_SIZE, -1) 
    grad_norm = torch.linalg.vector_norm(grad_average, ord=2, dim=1)

    gradient_penalty = torch.square(grad_norm - 1).mean()

    scores_f = discriminator(fake_samples) # N x 1
    scores_r = discriminator(real_samples) # N x 1

    # Discriminator objective:
    # Maximize the difference in critic scores for samples from the real distribution and generator's distribution

    discriminator_loss = - (scores_r.mean() - scores_f.mean()) + lambda_gp  * gradient_penalty
    return discriminator_loss

