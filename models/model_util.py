import torch
import torch.nn as nn

def save(net,path,gpu_id):
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.cpu().state_dict(),path)
    else:
        torch.save(net.cpu().state_dict(),path) 
    if gpu_id != '-1':
        net.cuda()