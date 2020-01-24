import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
import random
import sys
sys.path.append("..")
import util.image_processing as impro
from util import util,mosaic
import datetime
import shutil

mask_dir = '/media/hypo/Project/MyProject/DeepMosaics/DeepMosaics/train/add/datasets/av/mask'
img_dir ='/media/hypo/Project/MyProject/DeepMosaics/DeepMosaics/train/add/datasets/av/origin_image'
output_dir = './datasets_img'
util.makedirs(output_dir)
HD = True # if false make dataset for pix2pix, if Ture for pix2pix_HD
MASK = True # if True, output mask,too
OUT_SIZE = 256
FOLD_NUM = 2
Bounding = True

if HD:
    train_A_path = os.path.join(output_dir,'train_A')
    train_B_path = os.path.join(output_dir,'train_B')
    util.makedirs(train_A_path)
    util.makedirs(train_B_path)
else:
    train_path = os.path.join(output_dir,'train')
    util.makedirs(train_path)
if MASK:
    mask_path = os.path.join(output_dir,'mask')
    util.makedirs(mask_path)

mask_names = os.listdir(mask_dir)
img_names = os.listdir(img_dir)
mask_names.sort()
img_names.sort()
print('Find images:',len(img_names))

cnt = 0
for fold in range(FOLD_NUM):
    for img_name,mask_name in zip(img_names,mask_names):
        try:
            img = impro.imread(os.path.join(img_dir,img_name))
            mask = impro.imread(os.path.join(mask_dir,mask_name),'gray')
            mask = impro.resize_like(mask, img)
            x,y,size,area = impro.boundingSquare(mask, 1.5)
            if area > 100:
                if Bounding：
                    img = impro.resize(img[y-size:y+size,x-size:x+size],OUT_SIZE) 
                    mask =  impro.resize(mask[y-size:y+size,x-size:x+size],OUT_SIZE)
                img_mosaic = mosaic.addmosaic_random(img, mask)

                if HD:
                    cv2.imwrite(os.path.join(train_A_path,'%05d' % cnt+'.jpg'), img_mosaic)
                    cv2.imwrite(os.path.join(train_B_path,'%05d' % cnt+'.jpg'), img)
                else:
                    merge_img = impro.makedataset(img_mosaic, img)
                    cv2.imwrite(os.path.join(train_path,'%05d' % cnt+'.jpg'), merge_img)
                if MASK:
                    cv2.imwrite(os.path.join(mask_path,'%05d' % cnt+'.png'), mask)
                print("Processing:",img_name," ","Remain:",len(img_names)*FOLD_NUM-cnt)
                
        except Exception as e:
            print(img_name,e)
        cnt += 1
