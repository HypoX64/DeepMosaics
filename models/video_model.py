import torch
import torch.nn as nn
import torch.nn.functional as F
from .pix2pix_model import *


class encoder_2d(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(encoder_2d, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        #torch.Size([1, 256, 32, 32])

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class decoder_2d(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(decoder_2d, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        #torch.Size([1, 256, 32, 32])

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            #https://distill.pub/2016/deconv-checkerboard/
            #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190

            model += [  nn.Upsample(scale_factor = 2, mode='nearest'),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]
        # model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)



class conv_3d(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=3,stride=2,padding=1,norm_layer_3d=nn.BatchNorm3d,use_bias=True):
        super(conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
            norm_layer_3d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_2d(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=3,stride=1,padding=1,norm_layer_2d=nn.BatchNorm2d,use_bias=True):
        super(conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=0, bias=use_bias),
            norm_layer_2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class encoder_3d(nn.Module):
    def __init__(self,in_channel,norm_layer_2d,norm_layer_3d,use_bias):
        super(encoder_3d, self).__init__()
        self.down1 = conv_3d(1, 64, 3, 2, 1,norm_layer_3d,use_bias)
        self.down2 = conv_3d(64, 128, 3, 2, 1,norm_layer_3d,use_bias)
        self.down3 = conv_3d(128, 256, 3, 1, 1,norm_layer_3d,use_bias)
        self.conver2d = nn.Sequential(
            nn.Conv2d(256*int(in_channel/4), 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer_2d(256),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        x = x.view(x.size(0),1,x.size(1),x.size(2),x.size(3))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = x.view(x.size(0),x.size(1)*x.size(2),x.size(3),x.size(4))

        x = self.conver2d(x)

        return x



class ALL(nn.Module):
    def __init__(self, in_channel, out_channel,norm_layer_2d,norm_layer_3d,use_bias):
        super(ALL, self).__init__()

        self.encoder_2d = encoder_2d(4,-1,64,norm_layer=norm_layer_2d,n_blocks=9)
        self.encoder_3d = encoder_3d(in_channel,norm_layer_2d,norm_layer_3d,use_bias)
        self.decoder_2d = decoder_2d(4,3,64,norm_layer=norm_layer_2d,n_blocks=9)
        self.shortcut_cov = conv_2d(3,64,7,1,3,norm_layer_2d,use_bias)
        self.merge1 = conv_2d(512,256,3,1,1,norm_layer_2d,use_bias)
        self.merge2 = nn.Sequential(
            conv_2d(128,64,3,1,1,norm_layer_2d,use_bias),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channel, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):

        N = int((x.size()[1])/3)
        x_2d = torch.cat((x[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], x[:,N-1:N,:,:]), 1)
        shortcut_2d = x[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:]

        x_2d = self.encoder_2d(x_2d)

        x_3d = self.encoder_3d(x)
        x = torch.cat((x_2d,x_3d),1)
        x = self.merge1(x)
        x = self.decoder_2d(x)
        shortcut_2d = self.shortcut_cov(shortcut_2d)
        x = torch.cat((x,shortcut_2d),1)
        x = self.merge2(x)

        return x

def MosaicNet(in_channel, out_channel, norm='batch'):

    if norm == 'batch':
        # norm_layer_2d = nn.BatchNorm2d
        # norm_layer_3d = nn.BatchNorm3d
        norm_layer_2d = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        norm_layer_3d = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
        use_bias = False
    elif norm == 'instance':
        norm_layer_2d = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        norm_layer_3d = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
        use_bias = True

    return ALL(in_channel, out_channel, norm_layer_2d, norm_layer_3d, use_bias)
