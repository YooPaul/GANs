import torch
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.utils.tensorboard as tensorboard
import torchvision.datasets as datasets

from models import Generator, Discriminator
from util import init_weights, clip_weights

EPOCHS = 5
BATCH_SIZE = 64
MODEL_DATA = "Models/WGAN_model.pt"
DATASET_ROOT = 'Datasets/img_align_celeba'

lr = 5e-5
z_dim = 100
image_channels = 3
image_size = 64
n_critic = 5
c = 0.01 # Discriminator weights clipping parameter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

fixed_z = torch.rand((64, z_dim, 1, 1)).to(device) # for visualizing generator output throughout training

# Set up tensorboard writers
r_summary_writer = tensorboard.SummaryWriter('Tensorboard/WGAN/logs/real')
f_summary_writer = tensorboard.SummaryWriter('Tensorboard/WGAN/logs/fake')
#real_summary_writer = summary.create_file_writer('logs/real/')
#fake_summary_writer = summary.create_file_writer('logs/fake/')

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(), #  normalizes image pixel values to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]) #mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#train_dataset = datasets.MNIST(DATASET_ROOT, train=True, transform=preprocess, download=True)
train_dataset = CelebA(DATASET_ROOT, preprocess)
train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(z_dim, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)

# Initialize model weights from a normal distribution
generator.apply(init_weights)
discriminator.apply(init_weights)

optim_g = torch.optim.RMSprop(generator.parameters(), lr=lr)
optim_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
step = 0

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
        for _ in range(n_critic):
            idx, real_samples = out
            real_samples = real_samples.to(device)

            z = torch.rand((BATCH_SIZE, z_dim, 1, 1)).to(device)
            fake_samples = generator(z)
            
            scores_f = discriminator(fake_samples) # N x 1
            scores_r = discriminator(real_samples) # N x 1

            # Discriminator objective:
            # Maximize the difference in critic scores for samples from the real distribution and generator's distribution
            optim_d.zero_grad()
            discriminator_loss = - (scores_r.mean() - scores_f.mean())
            discriminator_loss.backward()#retain_graph=True) # need to retain graph in order to backpropogate through preds_f again later 
            
            optim_d.step()

            # discriminator weight clipping
            discriminator.apply(clip_weights)
            
            out = next(it, None)
            if out is None:
                break

        # Generator objective:
        # Maximize critic score given samples from the fake distribution
        z = torch.rand((BATCH_SIZE, z_dim, 1, 1)).to(device)
        fake_samples = generator(z)
        optim_g.zero_grad()
        scores_f = discriminator(fake_samples)
        generator_loss = -scores_f.mean()
        generator_loss.backward()

        optim_g.step()

        step += 1
        # Print loss and save model every 30 iterations
        if (step - 1) % 3 == 0:
            f_summary_writer.add_scalar('Generator loss', generator_loss.item(), global_step=step)
            r_summary_writer.add_scalar('Discriminator  loss', discriminator_loss.item(), global_step=step)
            '''
            print()
            print('Step:', step)
            print('Generator Loss:', generator_loss.item())
            print('Discriminator Loss:', discriminator_loss.item())
            '''
        
            '''
            torch.save(the_model.state_dict(), PATH)

            the_model = TheModelClass(*args, **kwargs)
            the_model.load_state_dict(torch.load(PATH))
            '''
        
        if (step - 1) % 100 == 0:
            print('Epoch: %d/%d\tBatch: %04d/%d' % (epoch, EPOCHS, idx, len(train)))
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

                #with real_summary_writer.as_default():
                #    summary.image('real', img_grid_real, step=step)


        


