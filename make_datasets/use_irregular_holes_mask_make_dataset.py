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

ir_mask_path = './Irregular_Holes_mask'
img_dir ='/media/hypo/Hypoyun/Datasets/other/face512' 
MOD = 'mosaic' #HD | pix2pix | mosaic
MASK = False # if True, output mask,too
BOUNDING = False # if true the mosaic size will be more big
suffix = '_1'
output_dir = os.path.join('./datasets_img',MOD)
util.makedirs(output_dir)

if MOD == 'HD':
    train_A_path = os.path.join(output_dir,'train_A')
    train_B_path = os.path.join(output_dir,'train_B')
    util.makedirs(train_A_path)
    util.makedirs(train_B_path)
elif MOD == 'pix2pix':
    train_path = os.path.join(output_dir,'train')
    util.makedirs(train_path)
elif MOD == 'mosaic':
    ori_path = os.path.join(output_dir,'ori')
    mosaic_path = os.path.join(output_dir,'mosaic')
    mask_path = os.path.join(output_dir,'mask')
    util.makedirs(ori_path)
    util.makedirs(mosaic_path)
    util.makedirs(mask_path)
if MASK:
    mask_path = os.path.join(output_dir,'mask')
    util.makedirs(mask_path)

transform_mask = transforms.Compose([
     transforms.RandomResizedCrop(size=512, scale=(0.5,1)),
     transforms.RandomHorizontalFlip(),
 ])

transform_img = transforms.Compose([

     transforms.Resize(512),
     transforms.RandomCrop(512)
 ])

mask_names = os.listdir(ir_mask_path)
img_paths = util.Traversal(img_dir)
img_paths = util.is_imgs(img_paths)
print('Find images:',len(img_paths))

for i,img_path in enumerate(img_paths,1):
    try:        
        img = Image.open(img_path)
        img = transform_img(img)
        img = np.array(img)
        img = img[...,::-1]

        if BOUNDING:
            mosaic_area = 0
            while mosaic_area < 16384:
                mask = Image.open(os.path.join(ir_mask_path,random.choices(mask_names)[0]))
                mask = transform_mask(mask)
                mask = np.array(mask)
                mosaic_area = impro.mask_area(mask)
            mosaic_img = mosaic.addmosaic_random(img, mask,'bounding') 
        else:
            mask = Image.open(os.path.join(ir_mask_path,random.choices(mask_names)[0]))
            mask = transform_mask(mask)
            mask = np.array(mask)
            mosaic_img = mosaic.addmosaic_random(img, mask)
                 
        if MOD == 'HD':#[128:384,128:384,:] --->256
            cv2.imwrite(os.path.join(train_A_path,'%05d' % i+suffix+'.jpg'), mosaic_img)
            cv2.imwrite(os.path.join(train_B_path,'%05d' % i+suffix+'.jpg'), img)
            if MASK:
                cv2.imwrite(os.path.join(mask_path,'%05d' % i+suffix+'.png'), mask)
        elif MOD == 'pix2pix':
            merge_img = impro.makedataset(mosaic_img, img)
            cv2.imwrite(os.path.join(train_path,'%05d' % i+suffix+'.jpg'), merge_img)
        elif MOD == 'mosaic':
            cv2.imwrite(os.path.join(mosaic_path,'%05d' % i+suffix+'.jpg'), mosaic_img)
            cv2.imwrite(os.path.join(ori_path,'%05d' % i+suffix+'.jpg'), img)
            cv2.imwrite(os.path.join(mask_path,'%05d' % i+suffix+'.png'), mask)

        print('\r','Proc/all:'+str(i)+'/'+str(len(img_paths)),util.get_bar(100*i/len(img_paths),num=40),end='')
    except Exception as e:
        print(img_path,e)
