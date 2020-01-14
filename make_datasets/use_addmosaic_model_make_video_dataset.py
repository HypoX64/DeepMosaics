import os
import numpy as np
import cv2
import random

import sys
sys.path.append("..")
from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg,filt
from util import image_processing as impro
from cores import options

opt = options.Options().getparse()
util.file_init(opt)

videos = os.listdir('./video')
videos.sort()
opt.model_path = '../pretrained_models/add_youknow_128.pth'
opt.use_gpu = True
Ex = 1.4
Area_Type  = 'normal'
suffix = ''

net = loadmodel.unet(opt)
for i,path in enumerate(videos,0):
    try:
        path = os.path.join('./video',path)
        util.clean_tempfiles()
        ffmpeg.video2voice(path,'./tmp/voice_tmp.mp3')
        ffmpeg.video2image(path,'./tmp/video2image/output_%05d.'+opt.tempimage_type)
        imagepaths=os.listdir('./tmp/video2image')
        imagepaths.sort()

        # get position
        positions = []
        img_ori_example = impro.imread(os.path.join('./tmp/video2image',imagepaths[0]))
        mask_avg = np.zeros((impro.resize(img_ori_example, 128)).shape[:2])
        for imagepath in imagepaths:
            imagepath = os.path.join('./tmp/video2image',imagepath)
            #print('Find ROI location:',imagepath)
            img = impro.imread(imagepath)
            x,y,size,mask = runmodel.get_mosaic_position(img,net,opt,threshold = 80)
            cv2.imwrite(os.path.join('./tmp/ROI_mask',
                              os.path.basename(imagepath)),mask)
            positions.append([x,y,size])
            mask_avg = mask_avg + mask
        #print('Optimize ROI locations...')
        mask_index = filt.position_medfilt(np.array(positions), 13)

        mask = np.clip(mask_avg/len(imagepaths),0,255).astype('uint8')
        mask = impro.mask_threshold(mask,20,32)
        x,y,size,area = impro.boundingSquare(mask,Ex_mul=Ex)
        rat = min(img_ori_example.shape[:2])/128.0
        x,y,size = int(rat*x),int(rat*y),int(rat*size)
        cv2.imwrite(os.path.join('./tmp/ROI_mask_check',
                                'test_show.png'),mask)
        if size !=0 :
            mask_path = './dataset/'+os.path.splitext(os.path.basename(path))[0]+suffix+'/mask'
            ori_path = './dataset/'+os.path.splitext(os.path.basename(path))[0]+suffix+'/ori'
            mosaic_path = './dataset/'+os.path.splitext(os.path.basename(path))[0]+suffix+'/mosaic'
            os.makedirs('./dataset/'+os.path.splitext(os.path.basename(path))[0]+suffix)
            os.makedirs(mask_path)
            os.makedirs(ori_path)
            os.makedirs(mosaic_path)
            #print('Add mosaic to images...')
            mosaic_size = mosaic.get_autosize(img_ori_example,mask,area_type = Area_Type)*random.uniform(1,2)
            models = ['squa_avg','rect_avg','squa_mid']
            mosaic_type = random.randint(0,len(models)-1)
            rect_rat = random.uniform(1.2,1.6)
            for i in range(len(imagepaths)):
                mask = impro.imread(os.path.join('./tmp/ROI_mask',imagepaths[mask_index[i]]),mod = 'gray')
                img_ori = impro.imread(os.path.join('./tmp/video2image',imagepaths[i]))
                img_mosaic = mosaic.addmosaic_normal(img_ori,mask,mosaic_size,model = models[mosaic_type],rect_rat=rect_rat)
                mask = impro.resize(mask, min(img_ori.shape[:2]))

                img_ori_crop = impro.resize(img_ori[y-size:y+size,x-size:x+size],256)
                img_mosaic_crop = impro.resize(img_mosaic[y-size:y+size,x-size:x+size],256)
                mask_crop = impro.resize(mask[y-size:y+size,x-size:x+size],256)

                cv2.imwrite(os.path.join(ori_path,os.path.basename(imagepaths[i])),img_ori_crop)
                cv2.imwrite(os.path.join(mosaic_path,os.path.basename(imagepaths[i])),img_mosaic_crop)
                cv2.imwrite(os.path.join(mask_path,os.path.basename(imagepaths[i])),mask_crop)
    except Exception as e:
        print(e)

    print(util.get_bar(100*i/len(videos),num=50))