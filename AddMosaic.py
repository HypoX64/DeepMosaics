import sys
import os
import random

import numpy as np
import cv2
import torch

from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg
from util import image_processing as impro
from options.addmosaic_options import AddOptions


opt = AddOptions().getparse()

#find mosaic position in image and add mosaic to this image
def add_mosaic_to_image(path):
    img = cv2.imread(path)
    mask =runmodel.run_unet_rectim(img,net,use_gpu = opt.use_gpu)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    img = mosaic.addmosaic(img,mask,opt.mosaic_size,opt.output_size,model = opt.mosaic_mod)
    return img

net = loadmodel.unet(os.path.join(opt.model_dir,opt.model_name),use_gpu = opt.use_gpu)

filepaths = util.Traversal(opt.input_dir)

for path in filepaths:
    if util.is_img(path):
        img = add_mosaic_to_image(path)
        cv2.imwrite(os.path.join(opt.result_dir,os.path.basename(path)),img)
    elif util.is_video(path):
        util.clean_tempfiles()
        fps = ffmpeg.get_video_infos(path)[0]
        ffmpeg.video2voice(path,'./tmp/voice_tmp.mp3')
        ffmpeg.video2image(path,'./tmp/video2image/output_%05d.png')
        for imagepath in os.listdir('./tmp/video2image'):
            imagepath = os.path.join('./tmp/video2image',imagepath)
            print(imagepath)
            img = add_mosaic_to_image(imagepath)
            cv2.imwrite(os.path.join('./tmp/addmosaic_image',
                                        os.path.basename(imagepath)),img)
        ffmpeg.image2video( fps,
                            './tmp/addmosaic_image/output_%05d.png',
                            './tmp/voice_tmp.mp3',
                             os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_AddMosaic.mp4'))

