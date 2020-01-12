import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


class conv_3d(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=3,stride=2,padding=1):
        super(conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class encoder_3d(nn.Module):
    def __init__(self,in_channel):
        super(encoder_3d, self).__init__()
        self.down1 = conv_3d(1, 64, 3, 2, 1)
        self.down2 = conv_3d(64, 128, 3, 2, 1)
        self.down3 = conv_3d(128, 256, 3, 2, 1)
        self.down4 = conv_3d(256, 512, 3, 2, 1)
        self.conver2d = nn.Sequential(
            nn.Conv2d(int(in_channel/16)+1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = x.view(x.size(0),1,x.size(1),x.size(2),x.size(3))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.view(x.size(1),x.size(2),x.size(3),x.size(4))
        x = self.conver2d(x)
        x = x.view(x.size(1),x.size(0),x.size(2),x.size(3))
        # print(x.size())
        # x = self.avgpool(x)
        return x




class encoder_2d(nn.Module):
    def __init__(self, in_channel):
        super(encoder_2d, self).__init__()
        self.inc = inconv(in_channel, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1,x2,x3,x4,x5

class decoder_2d(nn.Module):
    def __init__(self, out_channel):
        super(decoder_2d, self).__init__()
        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 64,bilinear=False)
        self.outc = outconv(64, out_channel)

    def forward(self,x5,x4,x3,x2,x1):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x


class HypoNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HypoNet, self).__init__()

        self.encoder_2d = encoder_2d(4)
        self.encoder_3d = encoder_3d(in_channel)
        self.decoder_2d = decoder_2d(out_channel)

    def forward(self, x):

        N = int((x.size()[1])/3)
        x_2d = torch.cat((x[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], x[:,N-1:N,:,:]), 1)
        # print(x_2d.size())
        x_3d = self.encoder_3d(x)

        x1,x2,x3,x4,x5 = self.encoder_2d(x_2d)
        x5 = x5 + x_3d
        x_2d = self.decoder_2d(x5,x4,x3,x2,x1)

        return x_2d

