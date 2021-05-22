import os
import time
import numpy as np
import cv2
from models import runmodel
from util import mosaic,util,ffmpeg,filt
from util import image_processing as impro
from .init import video_init

'''
---------------------Style Transfer---------------------
'''
def styletransfer_img(opt,netG):
    print('Style Transfer_img:',opt.media_path)
    img = impro.imread(opt.media_path)
    img = runmodel.run_styletransfer(opt, netG, img)
    suffix = os.path.basename(opt.model_path).replace('.pth','').replace('style_','')
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(opt.media_path))[0]+'_'+suffix+'.jpg'),img)

def styletransfer_video(opt,netG):
    path = opt.media_path
    fps,imagepaths = video_init(opt,path)[:2]
    print('Step:2/4 -- Transfer')
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
    length = len(imagepaths)

    for i,imagepath in enumerate(imagepaths,1):
        img = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        img = runmodel.run_styletransfer(opt, netG, img)
        cv2.imwrite(os.path.join(opt.temp_dir+'/style_transfer',imagepath),img)
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('preview',img)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i,len(imagepaths)),end='')
    
    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    suffix = os.path.basename(opt.model_path).replace('.pth','').replace('style_','')
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/style_transfer/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_'+suffix+'.mp4')) 