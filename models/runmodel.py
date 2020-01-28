import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch

def run_unet(img,net,size = 224,use_gpu = True):
    img=impro.image2folat(img,3)
    img=img.reshape(1,3,size,size)
    img = torch.from_numpy(img)
    if use_gpu:
        img=img.cuda()
    pred = net(img)
    pred = (pred.cpu().detach().numpy()*255)
    pred = pred.reshape(size,size).astype('uint8')
    return pred

def run_unet_rectim(img,net,size = 224,use_gpu = True):
    img = impro.resize(img,size)
    img1,img2 = impro.spiltimage(img,size)
    mask1 = run_unet(img1,net,size,use_gpu = use_gpu)
    mask2 = run_unet(img2,net,size,use_gpu = use_gpu)
    mask = impro.mergeimage(mask1,mask2,img,size)
    return mask

def run_pix2pix(img,net,opt):
    if opt.netG == 'HD':
        img = impro.resize(img,512)
    else:
        img = impro.resize(img,128)
    img = data.im2tensor(img,use_gpu=opt.use_gpu)
    img_fake = net(img)
    img_fake = data.tensor2im(img_fake)
    return img_fake

def run_styletransfer(opt, net, img, outsize = 720):
    if min(img.shape[:2]) >= outsize:
        img = impro.resize(img,outsize)
    img = img[0:4*int(img.shape[0]/4),0:4*int(img.shape[1]/4),:]
    img = data.im2tensor(img,use_gpu=opt.use_gpu)
    img = net(img)
    img = data.tensor2im(img)
    return img

def get_ROI_position(img,net,opt):
    mask = run_unet_rectim(img,net,use_gpu = opt.use_gpu)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    x,y,halfsize,area = impro.boundingSquare(mask, 1)
    return mask,x,y,area

def get_mosaic_position(img_origin,net_mosaic_pos,opt,threshold = 128 ):
    mask = run_unet_rectim(img_origin,net_mosaic_pos,use_gpu = opt.use_gpu)
    #mask_1 = mask.copy()
    mask = impro.mask_threshold(mask,30,threshold)
    mask = impro.find_best_ROI(mask)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=1.5)
    rat = min(img_origin.shape[:2])/224.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    return x,y,size,mask