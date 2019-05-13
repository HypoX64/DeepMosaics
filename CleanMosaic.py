import os
import random

import numpy as np
import cv2
import torch
import scipy.signal as signal

from models import runmodel,loadmodel
from util import util,ffmpeg,data
from util import image_processing as impro
from options.cleanmosaic_options import CleanOptions


opt = CleanOptions().getparse()

def get_mosaic_position(img_origin):
    mask =runmodel.run_unet_rectim(img_origin,net_mosaic_pos,use_gpu = opt.use_gpu)
    mask = impro.mask_threshold(mask,10,128)
    x,y,size,area = impro.boundingSquare(mask,Ex_mul=1.5)
    rat = min(img_origin.shape[:2])/128.0
    x,y,size = int(rat*x),int(rat*y),int(rat*size)
    return x,y,size

def replace_mosaic(img_origin,img_fake,x,y,size,no_father = opt.no_feather):
    img_fake = impro.resize(img_fake,size*2)

    if no_father:
        img_origin[y-size:y+size,x-size:x+size]=img_fake
        img_result = img_origin
    else:
        eclosion_num = int(size/5)
        entad = int(eclosion_num/2+2)
        mask = np.zeros(img_origin.shape, dtype='uint8')
        mask = cv2.rectangle(mask,(x-size+entad,y-size+entad),(x+size-entad,y+size-entad),(255,255,255),-1)
        mask = (cv2.blur(mask, (eclosion_num, eclosion_num)))
        mask = mask/255.0

        img_tmp = np.zeros(img_origin.shape)
        img_tmp[y-size:y+size,x-size:x+size]=img_fake
        img_result = img_origin.copy()
        img_result = (img_origin*(1-mask)+img_tmp*mask).astype('uint8')
    return img_result

netG = loadmodel.pix2pix(os.path.join(opt.model_dir,opt.model_name),opt.model_type_netG,use_gpu = opt.use_gpu)
net_mosaic_pos = loadmodel.unet(os.path.join(opt.model_dir,opt.mosaic_position_model_name),use_gpu = opt.use_gpu)

filepaths = util.Traversal(opt.input_dir)

for path in filepaths:
    if util.is_img(path):
        print('Clean Mosaic:',path)
        img_origin = cv2.imread(path)
        x,y,size = get_mosaic_position(img_origin)
        img_result = img_origin.copy()
        if size != 0 :
            img_mosaic = img_origin[y-size:y+size,x-size:x+size]
            img_fake=runmodel.run_pix2pix(img_mosaic,netG,use_gpu = opt.use_gpu)
            img_result = replace_mosaic(img_origin,img_fake,x,y,size)
        cv2.imwrite(os.path.join(opt.result_dir,os.path.basename(path)),img_result)

    elif util.is_video(path):
        util.clean_tempfiles()
        fps = ffmpeg.get_video_infos(path)[0]
        ffmpeg.video2voice(path,'./tmp/voice_tmp.mp3')
        ffmpeg.video2image(path,'./tmp/video2image/output_%05d.'+opt.tempimage_type)
        positions = []
        imagepaths=os.listdir('./tmp/video2image')
        imagepaths.sort()
        for imagepath in imagepaths:
            imagepath=os.path.join('./tmp/video2image',imagepath)
            img_origin = cv2.imread(imagepath)
            x,y,size = get_mosaic_position(img_origin)
            positions.append([x,y,size])
            print('Find Positions:',imagepath)
        
        positions =np.array(positions)
        for i in range(3):positions[:,i] =signal.medfilt(positions[:,i],opt.medfilt_num)

        for i,imagepath in enumerate(imagepaths,0):
            imagepath=os.path.join('./tmp/video2image',imagepath)
            x,y,size = positions[i][0],positions[i][1],positions[i][2]
            img_origin = cv2.imread(imagepath)
            img_result = img_origin.copy()
            if size != 0:
                img_mosaic = img_origin[y-size:y+size,x-size:x+size]
                img_fake=runmodel.run_pix2pix(img_mosaic,netG,use_gpu = opt.use_gpu)
                img_result = replace_mosaic(img_origin,img_fake,x,y,size)
            cv2.imwrite(os.path.join('./tmp/replace_mosaic',os.path.basename(imagepath)),img_result)
            print('Clean Mosaic:',imagepath)
        ffmpeg.image2video( fps,
                    './tmp/replace_mosaic/output_%05d.'+opt.tempimage_type,
                    './tmp/voice_tmp.mp3',
                     os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_CleanMosaic.mp4'))
