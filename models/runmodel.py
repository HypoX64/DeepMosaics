import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch
import numpy as np

def run_unet(img,net,size = 224,use_gpu = True):
    img = impro.resize(img,size)
    img = data.im2tensor(img,use_gpu = use_gpu,  bgr2rgb = False,use_transform = False , is0_1 = True)
    mask = net(img)
    mask = data.tensor2im(mask, gray=True,rgb2bgr = False, is0_1 = True)
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

def traditional_cleaner(img,opt):
    h,w = img.shape[:2]
    img = cv2.blur(img, (opt.tr_blur,opt.tr_blur))
    img = img[::opt.tr_down,::opt.tr_down,:]
    img = cv2.resize(img, (w,h),interpolation=cv2.INTER_LANCZOS4)
    return img

def run_styletransfer(opt, net, img):

    if opt.output_size != 0:
        if 'resize' in opt.preprocess and 'resize_scale_width' not in opt.preprocess:
            img = impro.resize(img,opt.output_size)
        elif 'resize_scale_width' in opt.preprocess:
            img = cv2.resize(img, (opt.output_size,opt.output_size))
        img = img[0:4*int(img.shape[0]/4),0:4*int(img.shape[1]/4),:]

    if 'edges' in opt.preprocess:
        if opt.canny > 100:
            canny_low = opt.canny-50
            canny_high = np.clip(opt.canny+50,0,255)
        elif opt.canny < 50:
            canny_low = np.clip(opt.canny-25,0,255)
            canny_high = opt.canny+25
        else:
            canny_low = opt.canny-int(opt.canny/2)
            canny_high = opt.canny+int(opt.canny/2)
        img = cv2.Canny(img,opt.canny-50,opt.canny+50)
        if opt.only_edges:
            return img
        img = data.im2tensor(img,use_gpu=opt.use_gpu,gray=True,use_transform = False,is0_1 = False)
    else:    
        img = data.im2tensor(img,use_gpu=opt.use_gpu,gray=False,use_transform = True)
    img = net(img)
    img = data.tensor2im(img)
    return img

def get_ROI_position(img,net,opt):
    mask = run_unet(img,net,size=224,use_gpu = opt.use_gpu)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    x,y,halfsize,area = impro.boundingSquare(mask, 1)
    return mask,x,y,area

def get_mosaic_position(img_origin,net_mosaic_pos,opt,threshold = 128 ):
    mask = run_unet(img_origin,net_mosaic_pos,size=224,use_gpu = opt.use_gpu)
    mask = impro.mask_threshold(mask,30,threshold)
    if not opt.all_mosaic_area:
        mask = impro.find_mostlikely_ROI(mask)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=opt.ex_mult)
    rat = min(img_origin.shape[:2])/224.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    return x,y,size,mask