import torch
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.utils.tensorboard as tensorboard
import torchvision.datasets as datasets

from models import Generator, Discriminator
from util import init_weights, wasserstein_loss_gp

EPOCHS = 30
BATCH_SIZE = 64
MODEL_DATA = "Models/WGAN_GP_model.pt"
DATASET_ROOT = 'Datasets/'

lr = 1e-4
z_dim = 100
image_channels = 3
image_size = 64
n_critic = 5
lambda_gp = 10

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

fixed_z = torch.rand((64, z_dim, 1, 1)).to(device) # for visualizing generator output throughout training

# Set up tensorboard writers
r_summary_writer = tensorboard.SummaryWriter('Tensorboard/WGAN_GP/logs/real')
f_summary_writer = tensorboard.SummaryWriter('Tensorboard/WGAN_GP/logs/fake')

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(), #  normalizes image pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(DATASET_ROOT, transform=preprocess, download=True)
train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

generator = Generator(z_dim, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)

# Initialize model weights from a normal distribution
generator.apply(init_weights)
discriminator.apply(init_weights)

optim_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0, 0.9))
optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0, 0.9))
step = 0

# If previous checkpoint exists, load model
if os.path.isfile(MODEL_DATA):
    checkpoint = torch.load(MODEL_DATA)
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    discriminator.load_state_dict(checkpoint['d_model_state_dict'])

    optim_g.load_state_dict(checkpoint['g_optimizer_state_dict'])
    optim_d.load_state_dict(checkpoint['d_optimizer_state_dict'])
    step = checkpoint['step']
    d_loss = checkpoint['d_loss']
    g_loss = checkpoint['g_loss']
    print('Previous step:', step)
    print('Previous generator loss', g_loss)
    print('Previous discriminator loss', d_loss)

    generator.train()
    discriminator.train()

for epoch in range(EPOCHS):
    it = iter(enumerate(train))
    out = next(it, None)
    while out != None:

        # Critic training loop
        for _ in range(n_critic):
            idx, (real_samples, _) = out
            real_samples = real_samples.to(device)

            z = torch.rand((BATCH_SIZE, z_dim, 1, 1)).to(device)
            fake_samples = generator(z)

            optim_d.zero_grad()
            discriminator_loss = wasserstein_loss_gp(real_samples, fake_samples, discriminator, device, lambda_gp)
            discriminator_loss.backward() # retain_graph=True) # need to retain graph in order to backpropogate through preds_f again later

            optim_d.step()

            out = next(it, None)
            if out is None:
                break

        # Generator training
        # Generator objective:
        # Maximize log probability of samples from the fake distribution
        z = torch.rand((BATCH_SIZE, z_dim, 1, 1)).to(device)
        fake_samples = generator(z)
        optim_g.zero_grad()
        scores_f = discriminator(fake_samples)
        generator_loss = -scores_f.mean()
        generator_loss.backward()

        optim_g.step()

        step += 1

        # Print loss and save model
        if (step - 1) % 3 == 0:
            f_summary_writer.add_scalar('Generator loss', generator_loss.item(), global_step=step)
            r_summary_writer.add_scalar('Discriminator  loss', discriminator_loss.item(), global_step=step)

        if (step - 1) % 100 == 0:
            print('Epoch: %d/%d\tBatch: %03d/%d' % (epoch, EPOCHS, idx, len(train)))
            torch.save({'step': step,
                        'g_model_state_dict': generator.state_dict(),
                        'g_optimizer_state_dict': optim_g.state_dict(),
                        'd_model_state_dict': discriminator.state_dict(),
                        'd_optimizer_state_dict': optim_d.state_dict(),
                        'g_loss': generator_loss.item(),
                        'd_loss': discriminator_loss.item(),
                        }, MODEL_DATA)

        if (step - 1) % 15 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_z)

                img_grid_real = utils.make_grid(real_samples[:64], normalize=True, value_range=(-1,1))

                img_grid_fake =  utils.make_grid(fake_samples, normalize=True, value_range=(-1,1))

                r_summary_writer.add_image('Real', img_grid_real, global_step=step)
                f_summary_writer.add_image('Fake', img_grid_fake, global_step=step)

