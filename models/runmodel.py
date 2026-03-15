import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch
import numpy as np

def _get_autocast_device(gpu_id):
    """Only enable autocast on CUDA — MPS float16 has stride/op issues."""
    if gpu_id != '-1' and gpu_id != 'mps':
        return 'cuda'
    return None

def run_segment(img,net,size = 360,gpu_id = '-1'):
    img = impro.resize(img,size)
    img = data.im2tensor(img,gpu_id = gpu_id, bgr2rgb = False, is0_1 = True)
    ac_dev = _get_autocast_device(gpu_id)
    with torch.no_grad():
        if ac_dev:
            with torch.autocast(ac_dev, dtype=torch.float16):
                mask = net(img)
        else:
            mask = net(img)
    mask = data.tensor2im(mask, gray=True, is0_1 = True)
    return mask

def run_segment_batch(imgs, net, size=360, gpu_id='-1'):
    """Process multiple images through segmentation in one batch."""
    resized = [impro.resize(img, size) for img in imgs]
    batch = np.stack([img.astype(np.float32) / 255.0 for img in resized])
    batch = batch.transpose((0, 3, 1, 2))
    batch_tensor = torch.from_numpy(batch).float()
    device = data._get_device(gpu_id)
    if device.type != 'cpu':
        batch_tensor = batch_tensor.to(device)
    ac_dev = _get_autocast_device(gpu_id)
    with torch.no_grad():
        if ac_dev:
            with torch.autocast(ac_dev, dtype=torch.float16):
                masks_tensor = net(batch_tensor)
        else:
            masks_tensor = net(batch_tensor)
    masks = []
    for i in range(masks_tensor.shape[0]):
        mask = data.tensor2im(masks_tensor, gray=True, is0_1=True, batch_index=i)
        masks.append(mask)
    return masks

def run_pix2pix(img,net,opt):
    if opt.netG == 'HD':
        img = impro.resize(img,512)
    else:
        img = impro.resize(img,128)
    img = data.im2tensor(img,gpu_id=opt.gpu_id)
    ac_dev = _get_autocast_device(opt.gpu_id)
    with torch.no_grad():
        if ac_dev:
            with torch.autocast(ac_dev, dtype=torch.float16):
                img_fake = net(img)
        else:
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
        img = cv2.Canny(img,canny_low,canny_high)
        if opt.only_edges:
            return img
        img = data.im2tensor(img,gpu_id=opt.gpu_id,gray=True)
    else:    
        img = data.im2tensor(img,gpu_id=opt.gpu_id)
    ac_dev = _get_autocast_device(opt.gpu_id)
    with torch.no_grad():
        if ac_dev:
            with torch.autocast(ac_dev, dtype=torch.float16):
                img = net(img)
        else:
            img = net(img)
    img = data.tensor2im(img)
    return img

def get_ROI_position(img,net,opt,keepsize=True):
    mask = run_segment(img,net,size=360,gpu_id = opt.gpu_id)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    if keepsize:
        mask = impro.resize_like(mask, img)
    x,y,halfsize,area = impro.boundingSquare(mask, 1)
    return mask,x,y,halfsize,area

def get_mosaic_position(img_origin,net_mosaic_pos,opt):
    h,w = img_origin.shape[:2]
    mask = run_segment(img_origin,net_mosaic_pos,size=360,gpu_id = opt.gpu_id)
    # mask_1 = mask.copy()
    mask = impro.mask_threshold(mask,ex_mun=int(min(h,w)/20),threshold=opt.mask_threshold)
    if not opt.all_mosaic_area:
        mask = impro.find_mostlikely_ROI(mask)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=opt.ex_mult)
    #Location fix
    rat = min(h,w)/360.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    x,y = np.clip(x, 0, w),np.clip(y, 0, h)
    size = np.clip(size, 0, min(w-x,h-y))
    # print(x,y,size)
    return x,y,size,mask