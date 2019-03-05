import sys
sys.path.append("..")
import util.image_processing as impro
from util import data
import torch


def run_unet(img,net,size = 128,use_gpu = True):
    img=impro.image2folat(img,3)
    img=img.reshape(1,3,size,size)
    img = torch.from_numpy(img)
    if use_gpu:
        img=img.cuda()
    pred = net(img)
    pred = (pred.cpu().detach().numpy()*255)
    pred = pred.reshape(size,size).astype('uint8')
    return pred

def run_unet_rectim(img,net,size = 128,use_gpu = True):
    img = impro.resize(img,size)
    img1,img2 = impro.spiltimage(img)
    mask1 = run_unet(img1,net,size = 128,use_gpu = use_gpu)
    mask2 = run_unet(img2,net,size = 128,use_gpu = use_gpu)
    mask = impro.mergeimage(mask1,mask2,img)
    return mask

def run_pix2pix(img,net,size = 128,use_gpu = True):
    img = impro.resize(img,size)
    img = data.im2tensor(img,use_gpu=use_gpu)
    img_fake = net(img)
    img_fake = data.tensor2im(img_fake)
    return img_fake
