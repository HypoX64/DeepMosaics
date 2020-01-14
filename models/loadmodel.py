import torch
from .pix2pix_model import define_G
from .pix2pixHD_model import define_G as define_G_HD
from .unet_model import UNet
from .video_model import MosaicNet

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')


def pix2pix(opt):
    # print(opt.model_path,opt.netG)
    if opt.netG == 'HD':
        netG = define_G_HD(3, 3, 64, 'global' ,4)
    else:
        netG = define_G(3, 3, 64, opt.netG, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG.eval()
    if opt.use_gpu:
        netG.cuda()
    return netG

def video(opt):
    netG = MosaicNet(3*25+1, 3)
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG.eval()
    if opt.use_gpu:
        netG.cuda()
    return netG


def unet_clean(opt):
    net = UNet(n_channels = 3, n_classes = 1)
    show_paramsnumber(net,'segment')
    net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net.eval()
    if opt.use_gpu:
        net.cuda()
    return net

def unet(opt):
    net = UNet(n_channels = 3, n_classes = 1)
    show_paramsnumber(net,'segment')
    net.load_state_dict(torch.load(opt.model_path))
    net.eval()
    if opt.use_gpu:
        net.cuda()
    return net
