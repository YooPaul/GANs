import torch
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.utils.tensorboard as tensorboard
import torchvision.datasets as datasets

from models import Generator, Discriminator
from util import init_weights

EPOCHS = 3
BATCH_SIZE = 128
MODEL_DATA = "DCGAN_model.pt"

lr = 2e-4
z_dim = 100
image_channels = 1
image_size = 64


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(), #  normalizes image pixel values to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]) #mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.MNIST('.', transform=preprocess, download=True)
train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(z_dim, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)

# Initialize model weights from a normal distribution
generator.apply(init_weights)
discriminator.apply(init_weights)

optim_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
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

    model.train()


for epoch in range(EPOCHS):
    for idx, (real_samples, _) in enumerate(train):
        real_samples = real_samples.to(device)

        z = torch.rand((BATCH_SIZE, z_dim, 1, 1)).to(device)
        fake_samples = generator(z)
        
        preds_f = discriminator(fake_samples)
        preds_r = discriminator(real_samples)

        # Discriminator objective:
        # Maximize log probability of samples from the real distribution
        # Minimize log probability of samples from the fake distribution
        optim_d.zero_grad()
        discriminator_loss = F.binary_cross_entropy(preds_r, torch.ones_like(preds_r)) + F.binary_cross_entropy(preds_f, torch.zeros_like(preds_f))
        discriminator_loss.backward(retain_graph=True) # need to retain graph in order to backpropogate through preds_f again later 
        
        optim_d.step()

        # Generator objective:
        # Maximize log probability of samples from the fake distribution
        optim_g.zero_grad()
        preds_f = discriminator(fake_samples)
        generator_loss = F.binary_cross_entropy(preds_f, torch.ones_like(preds_f))
        generator_loss.backward()

        optim_g.step()

        step += 1
        # Print loss and save model every 30 iterations
        if (step - 1) % 70 == 0:
            print()
            print('Step:', step)
            print('Generator Loss:', generator_loss.item())
            print('Discriminator Loss:', discriminator_loss.item())
            
            torch.save({'step': step,
                        'g_model_state_dict': generator.state_dict(),
                        'g_optimizer_state_dict': optim_g.state_dict(),
                        'd_model_state_dict': discriminator.state_dict(),
                        'd_optimizer_state_dict': optim_d.state_dict(),
                        'g_loss': generator_loss.item(),
                        'd_loss': discriminator_loss.item(),
                        }, MODEL_DATA)
            

        


