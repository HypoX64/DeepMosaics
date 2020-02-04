import numpy as np
import cv2
import os
import sys
sys.path.append("..")
from util import image_processing as impro
from util import util

img_dir = './datasets_img/pix2pix/edges2cat/images'
output_dir = './datasets_img/pix2pix/edges2cat/train'
util.makedirs(output_dir)

img_names = os.listdir(img_dir)
for i,img_name in enumerate(img_names,2000): 
    try:
        img = impro.imread(os.path.join(img_dir,img_name))
        img = impro.resize(img, 286)
        h,w = img.shape[:2]
        edges = cv2.Canny(img,150,250)
        edges = impro.ch_one2three(edges)
        out_img = np.zeros((h,w*2,3), dtype=np.uint8)
        out_img[:,0:w] = edges
        out_img[:,w:2*w] = img
        cv2.imwrite(os.path.join(output_dir,'%05d' % i+'.jpg'), out_img)
    except Exception as e:
        pass
