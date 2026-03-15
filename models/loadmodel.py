import torch
from . import model_util
from .pix2pix_model import define_G as pix2pix_G
from .pix2pixHD_model import define_G as pix2pixHD_G
from .BiSeNet_model import BiSeNet
from .BVDNet import define_G as video_G

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')

def _try_compile(net, gpu_id):
    if gpu_id == 'mps' or gpu_id == '-1':
        return net
    try:
        return torch.compile(net)
    except Exception:
        return net

def _finalize(net, gpu_id):
    net = model_util.todevice(net, gpu_id)
    net.eval()
    net = _try_compile(net, gpu_id)
    return net

def pix2pix(opt):
    if opt.netG == 'HD':
        netG = pix2pixHD_G(3, 3, 64, 'global' ,4)
    else:
        netG = pix2pix_G(3, 3, 64, opt.netG, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path, weights_only=False))
    return _finalize(netG, opt.gpu_id)


def style(opt):
    if opt.edges:
        netG = pix2pix_G(1, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
    else:
        netG = pix2pix_G(3, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=False, init_type='normal', gpu_ids=[])

    if isinstance(netG, torch.nn.DataParallel):
        netG = netG.module
    state_dict = torch.load(opt.model_path, map_location='cpu', weights_only=False)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    for key in list(state_dict.keys()):
        model_util.patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)

    return _finalize(netG, opt.gpu_id)

def video(opt):
    netG = video_G(N=2,n_blocks=4,gpu_id=opt.gpu_id)
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path, weights_only=False))
    return _finalize(netG, opt.gpu_id)

def bisenet(opt,type='roi'):
    '''
    type: roi or mosaic
    '''
    net = BiSeNet(num_classes=1, context_path='resnet18',train_flag=False)
    show_paramsnumber(net,'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path, weights_only=False))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path, weights_only=False))
    return _finalize(net, opt.gpu_id)
