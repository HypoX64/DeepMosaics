import sys
import os
import random

import numpy as np
import cv2
import torch

from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg
from util import image_processing as impro
from options import Options


opt = Options().getparse()
util.init(opt)

if opt.mode == 'add':

    net = loadmodel.unet(opt)
    path = opt.media_path
    if util.is_img(path):
        print('Add Mosaic:',path)
        img = cv2.imread(path)
        img = runmodel.add_mosaic_to_image(img,net,opt)
        cv2.imwrite(os.path.join(opt.result_dir,os.path.basename(path)),img)
    elif util.is_video(path):
        util.clean_tempfiles()
        fps = ffmpeg.get_video_infos(path)[0]
        ffmpeg.video2voice(path,'./tmp/voice_tmp.mp3')
        ffmpeg.video2image(path,'./tmp/video2image/output_%05d.'+opt.tempimage_type)
        for imagepath in os.listdir('./tmp/video2image'):
            imagepath = os.path.join('./tmp/video2image',imagepath)
            print('Add Mosaic:',imagepath)
            img = cv2.imread(imagepath)
            img = runmodel.add_mosaic_to_image(img,net,opt)
            cv2.imwrite(os.path.join('./tmp/addmosaic_image',
                                        os.path.basename(imagepath)),img)
        ffmpeg.image2video( fps,
                            './tmp/addmosaic_image/output_%05d.'+opt.tempimage_type,
                            './tmp/voice_tmp.mp3',
                             os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_AddMosaic.mp4'))

elif opt.mode == 'clean':
    netG = loadmodel.pix2pix(opt)
    net_mosaic_pos = loadmodel.unet_clean(opt)
    path = opt.media_path
    if util.is_img(path):
        print('Clean Mosaic:',path)
        img_origin = cv2.imread(path)
        x,y,size = runmodel.get_mosaic_position(img_origin,net_mosaic_pos,opt)
        img_result = img_origin.copy()
        if size != 0 :
            img_mosaic = img_origin[y-size:y+size,x-size:x+size]
            img_fake=runmodel.run_pix2pix(img_mosaic,netG,use_gpu = opt.use_gpu)
            img_result = impro.replace_mosaic(img_origin,img_fake,x,y,size,opt.no_feather)
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
            x,y,size = runmodel.get_mosaic_position(img_origin,net_mosaic_pos,opt)
            positions.append([x,y,size])
            print('Find Positions:',imagepath)
        
        positions =np.array(positions)
        for i in range(3):positions[:,i] = impro.medfilt(positions[:,i],opt.medfilt_num)

        for i,imagepath in enumerate(imagepaths,0):
            imagepath=os.path.join('./tmp/video2image',imagepath)
            x,y,size = positions[i][0],positions[i][1],positions[i][2]
            img_origin = cv2.imread(imagepath)
            img_result = img_origin.copy()
            if size != 0:
                img_mosaic = img_origin[y-size:y+size,x-size:x+size]
                img_fake = runmodel.run_pix2pix(img_mosaic,netG,use_gpu = opt.use_gpu)
                img_result = impro.replace_mosaic(img_origin,img_fake,x,y,size,opt.no_feather)
            cv2.imwrite(os.path.join('./tmp/replace_mosaic',os.path.basename(imagepath)),img_result)
            print('Clean Mosaic:',imagepath)
        ffmpeg.image2video( fps,
                    './tmp/replace_mosaic/output_%05d.'+opt.tempimage_type,
                    './tmp/voice_tmp.mp3',
                     os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_CleanMosaic.mp4'))                      

util.clean_tempfiles()