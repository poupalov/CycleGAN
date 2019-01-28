import torch.nn as nn

# Input images : 256 x 256

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ Basic residual block """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
 
    
class Generator(nn.Module):
    
    def __init__(self, nb_filters=64, act=True, batch_norm=False):
        super(Generator, self).__init__()
        
        # All shapes indications are done supposing input_size = 256, nb_filters = 64
        # However every power of 2 greater than 16 is a valid value for input_size
        # (assuming the generator outputs will be used by the discriminator)
        
        # Encoding
        encoder = []
        encoder.append(nn.Conv2d(3, nb_filters, kernel_size=7, stride=1, padding=3)) # Returns 256 x 256 x 64
        encoder.append(nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=3, stride=2, padding=1)) # Returns 128 x 128 x 128
        if batch_norm : encoder.append(nn.BatchNorm2d(nb_filters * 2))
        if act : encoder.append(nn.ReLU())
        encoder.append(nn.Conv2d(nb_filters * 2, nb_filters * 4, kernel_size=3, stride=2, padding=1)) # Returns 64 x 64 x 256
        if batch_norm : encoder.append(nn.BatchNorm2d(nb_filters * 4))
        if act : encoder.append(nn.ReLU())
        
        # Transformation
        transform = []
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256
        transform.append(BasicBlock(inplanes=nb_filters * 4, planes=nb_filters * 4)) # Returns 64 x 64 x 256

        # Decoding
        decoder = []
        decoder.append(nn.ConvTranspose2d(nb_filters * 4, nb_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)) # Returns 128 x 128 x 128
        if batch_norm : decoder.append(nn.BatchNorm2d(nb_filters * 2))
        if act : decoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(nb_filters * 2, nb_filters, kernel_size=3, stride=2, padding=1, output_padding=1)) # Returns 256 x 256 x 64
        if batch_norm : decoder.append(nn.BatchNorm2d(nb_filters))
        if act : decoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(nb_filters, 3, kernel_size=7, stride=1, padding=3)) # Returns 256 x 256 x 3
        decoder.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder)
        self.transform = nn.Sequential(*transform)
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.transform(x)
        x = self.decoder(x)
        return x
    
    
class Discriminator(nn.Module):
    
    def __init__(self, nb_filters=64, k=5, input_size=256):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        
        conv_layers = []
        p = int((k - 1) / 2) # padding
        conv_layers.append(nn.Conv2d(3, nb_filters, kernel_size=k, stride=2, padding=p))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(nb_filters, nb_filters * 2, kernel_size=k, stride=2, padding=p))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(nb_filters * 2, nb_filters * 4, kernel_size=k, stride=2, padding=p))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(nb_filters * 4, nb_filters * 8, kernel_size=k, stride=2, padding=p))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(nb_filters * 8, nb_filters * 1, kernel_size=k, stride=1, padding=int((k - 1) / 2)))
        
        decision_layers = []
        decision_layers.append(nn.Linear(int(nb_filters * (input_size / 16) ** 2), 1))
        decision_layers.append(nn.Sigmoid())
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.decision_layers = nn.Sequential(*decision_layers)
        self.nb_filters = nb_filters
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, int(self.nb_filters * (self.input_size / 16) ** 2)) 
        x = self.decision_layers(x)
        
        return x
    