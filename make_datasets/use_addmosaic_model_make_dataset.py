import sys
import os
import random
import datetime

import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from unet import UNet
from mosaic import random_mosaic
import image_processing as impro



def runmodel(img,net):
    img=impro.image2folat(img,3)
    img=img.reshape(1,3,128,128)
    img = torch.from_numpy(img)
    img=img.cuda()
    pred = net(img)
    pred = (pred.cpu().detach().numpy()*255)
    pred = pred.reshape(128,128).astype('uint8')
    return pred



dir_img = './origin_image/'
dir_mosaic = './mosaic/'
dir_mask = './mask/'
dir_dataset = './dataset/'
dir_checkpoint = 'checkpoints/'

net = UNet(n_channels = 3, n_classes = 1)
net.load_state_dict(torch.load(dir_checkpoint+'mosaic_position.pth'))
net.cuda()
net.eval()
# cudnn.benchmark = True
files = os.listdir(dir_mosaic)

for i,file in enumerate(files,1):
    orgin_image = cv2.imread(dir_img+file)
    mosaic_image = cv2.imread(dir_mosaic+file)
    img = impro.resize(mosaic_image,128)
    img1,img2 = impro.spiltimage(img)
    mask1 =runmodel(img1,net)
    mask2 =runmodel(img2,net)
    mask = impro.mergeimage(mask1,mask2,img)

    # test_mask = mask.copy()

    mask = impro.mask_threshold(mask,blur=5,threshold=128)
    if impro.mask_area(mask) > 1:
        h,w = orgin_image.shape[:2]
        mosaic_image = cv2.resize(mosaic_image,(w,h))
        # test_mask  = cv2.resize(test_mask,(w,h))
        # test_mask  = impro.ch_one2three(test_mask)

        x,y,size,area = impro.boundingSquare(mask,Ex_mul=1.5)
        rat = min(orgin_image.shape[:2])/128.0
        x,y,size = int(rat*x),int(rat*y),int(rat*size)
        orgin_crop = orgin_image[y-size:y+size,x-size:x+size]
        mosaic_crop = mosaic_image[y-size:y+size,x-size:x+size]
        # mosaic_crop = test_mask[y-size:y+size,x-size:x+size]

        result = impro.makedataset(mosaic_crop,orgin_crop)
        cv2.imwrite(dir_dataset+file,result)
    if i%1000==0:
        print(i,'image finished.')
