import torch
from .pix2pix_model import *
from .unet_model import UNet

def pix2pix(model_path,G_model_type,use_gpu = True):
    gpu_ids=[]
    if use_gpu:
        gpu_ids=[0]
    netG = define_G(3, 3, 64, G_model_type, norm='instance', init_type='normal', gpu_ids=gpu_ids)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()
    if use_gpu:
        netG.cuda()
    return netG

def unet(model_path,use_gpu = True):
    net = UNet(n_channels = 3, n_classes = 1)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    if use_gpu:
        net.cuda()
    return net


# def unet():