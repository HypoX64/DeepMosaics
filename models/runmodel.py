import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch
import numpy as np

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

def run_styletransfer(opt, net, img):
    if opt.output_size != 0:
        img = impro.resize(img,opt.output_size)
    if opt.edges:
        if not opt.only_edges:
            img = img[0:256*int(img.shape[0]/256),0:256*int(img.shape[1]/256),:]
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
    if not opt.no_large_area:
        mask = impro.find_best_ROI(mask)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=opt.ex_mult)
    rat = min(img_origin.shape[:2])/224.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    return x,y,size,mask