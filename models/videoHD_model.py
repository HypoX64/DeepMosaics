import torch
import torch.nn as nn
import torch.nn.functional as F
from .pix2pixHD_model import *


class encoder_2d(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(encoder_2d, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                      norm_layer(ngf * mult * 2), activation]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)  

class decoder_2d(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(decoder_2d, self).__init__()        
        activation = nn.ReLU(True)        

        model = []

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)

            # model += [  nn.Upsample(scale_factor = 2, mode='nearest'),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0),
            # norm_layer(int(ngf * mult / 2)),
            # nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
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
        self.inconv = conv_3d(1, 64, 7, 2, 3,norm_layer_3d,use_bias)
        self.down1 = conv_3d(64, 128, 3, 2, 1,norm_layer_3d,use_bias)
        self.down2 = conv_3d(128, 256, 3, 2, 1,norm_layer_3d,use_bias)
        self.down3 = conv_3d(256, 512, 3, 2, 1,norm_layer_3d,use_bias)
        self.down4 = conv_3d(512, 1024, 3, 1, 1,norm_layer_3d,use_bias)
        self.pool = nn.AvgPool3d((5,1,1))
        # self.conver2d = nn.Sequential(
        #     nn.Conv2d(256*int(in_channel/4), 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #     norm_layer_2d(256),
        #     nn.ReLU(inplace=True),
        # )


    def forward(self, x):

        x = x.view(x.size(0),1,x.size(1),x.size(2),x.size(3))
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        #print(x.size())
        x = self.pool(x)
        #print(x.size())
        # torch.Size([1, 1024, 16, 16])
        # torch.Size([1, 512, 5, 16, 16])


        x = x.view(x.size(0),x.size(1),x.size(3),x.size(4))

       # x = self.conver2d(x)

        return x

    # def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
    #              padding_type='reflect')

class ALL(nn.Module):
    def __init__(self, in_channel, out_channel,norm_layer_2d,norm_layer_3d,use_bias):
        super(ALL, self).__init__()

        self.encoder_2d = encoder_2d(4,3,64,4,norm_layer=norm_layer_2d,padding_type='reflect')
        self.encoder_3d = encoder_3d(in_channel,norm_layer_2d,norm_layer_3d,use_bias)
        self.decoder_2d = decoder_2d(4,3,64,4,norm_layer=norm_layer_2d,padding_type='reflect')
        # self.shortcut_cov = conv_2d(3,64,7,1,3,norm_layer_2d,use_bias)
        self.merge1 = conv_2d(2048,1024,3,1,1,norm_layer_2d,use_bias)
        # self.merge2 = nn.Sequential(
        #     conv_2d(128,64,3,1,1,norm_layer_2d,use_bias),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(64, out_channel, kernel_size=7, padding=0),
        #     nn.Tanh()
        # )

    def forward(self, x):

        N = int((x.size()[1])/3)
        x_2d = torch.cat((x[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], x[:,N-1:N,:,:]), 1)
        #shortcut_2d = x[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:]

        x_2d = self.encoder_2d(x_2d)
        x_3d = self.encoder_3d(x)
        #x = x_2d + x_3d
        x = torch.cat((x_2d,x_3d),1)
        x = self.merge1(x)

        x = self.decoder_2d(x)
        #shortcut_2d = self.shortcut_cov(shortcut_2d)
        #x = torch.cat((x,shortcut_2d),1)
        #x = self.merge2(x)

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
