import torch
from .pix2pix_model import define_G
from .pix2pixHD_model import define_G as define_G_HD
from .unet_model import UNet
from .video_model import MosaicNet
from .videoHD_model import MosaicNet as MosaicNet_HD
from .BiSeNet_model import BiSeNet

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def pix2pix(opt):
    # print(opt.model_path,opt.netG)
    if opt.netG == 'HD':
        netG = define_G_HD(3, 3, 64, 'global' ,4)
    else:
        netG = define_G(3, 3, 64, opt.netG, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG.eval()
    if opt.use_gpu != -1:
        netG.cuda()
    return netG


def style(opt):
    if opt.edges:
        netG = define_G(1, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
    else:
        netG = define_G(3, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=False, init_type='normal', gpu_ids=[])

    #in other to load old pretrain model
    #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/models/base_model.py
    if isinstance(netG, torch.nn.DataParallel):
        netG = netG.module
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(opt.model_path, map_location='cpu')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)

    if opt.use_gpu != -1:
        netG.cuda()
    return netG

def video(opt):
    if 'HD' in opt.model_path:
        netG = MosaicNet_HD(3*25+1, 3, norm='instance')
    else:
        netG = MosaicNet(3*25+1, 3,norm = 'batch')
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG.eval()
    if opt.use_gpu != -1:
        netG.cuda()
    return netG

def bisenet(opt,type='roi'):
    '''
    type: roi or mosaic
    '''
    net = BiSeNet(num_classes=1, context_path='resnet18',train_flag=False)
    show_paramsnumber(net,'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net.eval()
    if opt.use_gpu != -1:
        net.cuda()
    return net
