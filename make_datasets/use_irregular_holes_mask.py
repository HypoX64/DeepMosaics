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
# img_path = 'D:/MyProject_new/face_512'
img_path ='/media/hypo/Hypoyun/Hypoyun/手机摄影/20190219'
output_dir = './datasets'
util.makedirs(output_dir)
HD = True #if false make dataset for pix2pix, if Ture for pix2pix_HD
MASK = True
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

        mask = Image.open(os.path.join(ir_mask_path,random.choices(mask_names)[0]))
        mask = transform_mask(mask)
        mask = np.array(mask)

        mosaic_img = mosaic.addmosaic_random(img, mask)          
        if HD:
            cv2.imwrite(os.path.join(train_A_path,'%05d' % i+'.jpg'), mosaic_img)
            cv2.imwrite(os.path.join(train_B_path,'%05d' % i+'.jpg'), img)
        else:
            merge_img = impro.makedataset(mosaic_img, img)
            cv2.imwrite(os.path.join(train_path,'%05d' % i+'.jpg'), merge_img)
        if MASK:
            cv2.imwrite(os.path.join(mask_path,'%05d' % i+'.png'), mask)
        print("Processing:",img_name," ","Remain:",len(img_names)-i)
    except Exception as e:
        print(img_name,e)
