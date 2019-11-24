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
img_path ='/home/hypo/桌面/FaceRankSample' 
output_dir = './datasets_img'
util.makedirs(output_dir)
MOD = 'HD' #HD | pix2pix | mosaic
MASK = False # if True, output mask,too
BOUNDING = False # if true the mosaic size will be more big

if MOD='HD':
    train_A_path = os.path.join(output_dir,'train_A')
    train_B_path = os.path.join(output_dir,'train_B')
    util.makedirs(train_A_path)
    util.makedirs(train_B_path)
elif MOD='pix2pix':
    train_path = os.path.join(output_dir,'train')
    util.makedirs(train_path)
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
img_names = os.listdir(img_path)
print('Find images:',len(img_names))

for i,img_name in enumerate(img_names,1):
    try:
        img = Image.open(os.path.join(img_path,img_name))
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
            BOUNDING_flag = '_bound'
        else:
            mask = Image.open(os.path.join(ir_mask_path,random.choices(mask_names)[0]))
            mask = transform_mask(mask)
            mask = np.array(mask)
            mosaic_img = mosaic.addmosaic_random(img, mask)
            BOUNDING_flag = ''    
        
        if HD:#[128:384,128:384,:] --->256
            cv2.imwrite(os.path.join(train_A_path,'%05d' % i+BOUNDING_flag+'.jpg'), mosaic_img)
            cv2.imwrite(os.path.join(train_B_path,'%05d' % i+BOUNDING_flag+'.jpg'), img)
        else:
            merge_img = impro.makedataset(mosaic_img, img)
            cv2.imwrite(os.path.join(train_path,'%05d' % i+BOUNDING_flag+'.jpg'), merge_img)
        if MASK:
            cv2.imwrite(os.path.join(mask_path,'%05d' % i+BOUNDING_flag+'.png'), mask)
        print("Processing:",img_name," ","Remain:",len(img_names)-i)
    except Exception as e:
        print(img_name,e)
