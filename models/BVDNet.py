import torch
import torch.nn as nn
import torch.nn.functional as F
from .pix2pixHD_model import *
from .model_util import *


class Encoder2d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, activation = nn.ReLU(True)):
        super(Encoder2d, self).__init__()        
   
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0),
                      norm_layer(ngf * mult * 2), activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Encoder3d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm3d,activation = nn.ReLU(True)):
        super(Encoder3d, self).__init__()        
               
        model = [nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class BVDNet(nn.Module):
    def __init__(self, N, n_downsampling=3, n_blocks=1, input_nc=3, output_nc=3,norm='batch',activation=nn.LeakyReLU(0.2)):
        super(BVDNet, self).__init__()

        ngf = 64
        padding_type = 'reflect'
        norm_layer = get_norm_layer(norm,'2d')
        norm_layer_3d = get_norm_layer(norm,'3d')
        self.N = N

        # encoder
        self.encoder3d = Encoder3d(input_nc,64,n_downsampling,norm_layer_3d,activation)
        self.encoder2d = Encoder2d(input_nc,64,n_downsampling,norm_layer,activation)

        ### resnet blocks
        self.blocks = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nn.ReLU(True), norm_layer=norm_layer)]
        self.blocks = nn.Sequential(*self.blocks)

        ### decoder
        self.decoder = []        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
            # self.decoder += [   nn.Upsample(scale_factor = 2, mode='nearest'),
            #                     nn.ReflectionPad2d(1),
            #                     nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0),
            #                     norm_layer(int(ngf * mult / 2)),
            #                     activation]
        self.decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]        
        self.decoder = nn.Sequential(*self.decoder)
        self.limiter = nn.Tanh()

    def forward(self, stream, previous):
        this_shortcut = stream[:,:,self.N]
        stream = self.encoder3d(stream)
        stream = stream.reshape(stream.size(0),stream.size(1),stream.size(3),stream.size(4))
        # print(stream.shape)
        previous = self.encoder2d(previous)
        x = stream + previous
        x = self.blocks(x)
        x = self.decoder(x)
        x = x+this_shortcut
        x = self.limiter(x)
        #print(x.shape)

        # print(stream.shape,previous.shape)
        return x

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()  

        self.vgg = Vgg19()
        if gpu_ids != '-1' and len(gpu_ids) == 1:
            self.vgg.cuda()
        elif gpu_ids != '-1' and len(gpu_ids) > 1:
            self.vgg = nn.DataParallel(self.vgg)
            self.vgg.cuda()

        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
