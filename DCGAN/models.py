import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, output_c):
        super(Generator, self).__init__()

        self.dimensions = [1024, 512, 256, 128]

        # The paper specifices a kernel size of 5x5, however, that does not lead to an output image of 64x64
        layers = [self._deconv_block(z_dim, self.dimensions[0], 4, 1, 0)]
        for i in range(1, len(self.dimensions)):
            layers.append( self._deconv_block(self.dimensions[i - 1], self.dimensions[i], 4,  2,  1) )
        
        layers += [nn.Sequential(nn.ConvTranspose2d(self.dimensions[-1], output_c, 4, 2, 1),
                                 nn.Tanh())]

        self.layers = nn.ModuleList(layers)

    def _deconv_block(self, in_c, out_c, k_size, stride, pad):
        return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, k_size, stride, pad, bias=False), # no need to add bias due to BatchNorm right afterwards
                             nn.BatchNorm2d(out_c),
                             nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_c):
        super(Discriminator, self).__init__()

        self.dimensions = [64, 128, 256, 512]

        layers = [nn.Sequential(nn.Conv2d(input_c, self.dimensions[0], 4, 2, 1),
                                nn.LeakyReLU(0.2))]

        for i in range(1, len(self.dimensions)):
            layers.append( self._conv_block(self.dimensions[i - 1], self.dimensions[i], 4,  2,  1) )
        
        layers += [nn.Sequential(nn.Conv2d(self.dimensions[-1], 1, 4, 2, 0),
                                 nn.Sigmoid())]

        self.layers = nn.ModuleList(layers)

    def _conv_block(self, in_c, out_c, k_size, stride, pad):
        return nn.Sequential(nn.Conv2d(in_c, out_c, k_size, stride, pad, bias=False),
                             nn.BatchNorm2d(out_c),
                             nn.LeakyReLU(0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 1) # output size is N x 1
