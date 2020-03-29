import os
import numpy as np
import cv2

from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro

'''
---------------------Video Init---------------------
'''
def video_init(opt,path):
    util.clean_tempfiles()
    fps,endtime,height,width = ffmpeg.get_video_infos(path)
    if opt.fps !=0:
        fps = opt.fps
    ffmpeg.video2voice(path,'./tmp/voice_tmp.mp3')
    ffmpeg.video2image(path,'./tmp/video2image/output_%05d.'+opt.tempimage_type,fps)
    imagepaths=os.listdir('./tmp/video2image')
    imagepaths.sort()
    return fps,imagepaths,height,width

'''
---------------------Add Mosaic---------------------
'''
def addmosaic_img(opt,netS):
    path = opt.media_path
    print('Add Mosaic:',path)
    img = impro.imread(path)
    mask = runmodel.get_ROI_position(img,netS,opt)[0]
    img = mosaic.addmosaic(img,mask,opt)
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_add.jpg'),img)

def addmosaic_video(opt,netS):
    path = opt.media_path
    fps,imagepaths = video_init(opt,path)[:2]
    # get position
    positions = []
    for i,imagepath in enumerate(imagepaths,1):
        img = impro.imread(os.path.join('./tmp/video2image',imagepath))
        mask,x,y,area = runmodel.get_ROI_position(img,netS,opt)
        positions.append([x,y,area])      
        cv2.imwrite(os.path.join('./tmp/ROI_mask',imagepath),mask)
        print('\r','Find ROI location:'+str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print('\nOptimize ROI locations...')
    mask_index = filt.position_medfilt(np.array(positions), 7)

    # add mosaic
    for i in range(len(imagepaths)):
        mask = impro.imread(os.path.join('./tmp/ROI_mask',imagepaths[mask_index[i]]),'gray')
        img = impro.imread(os.path.join('./tmp/video2image',imagepaths[i]))
        if impro.mask_area(mask)>100:    
            img = mosaic.addmosaic(img, mask, opt)
        cv2.imwrite(os.path.join('./tmp/addmosaic_image',imagepaths[i]),img)
        print('\r','Add Mosaic:'+str(i+1)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print()
    ffmpeg.image2video( fps,
                        './tmp/addmosaic_image/output_%05d.'+opt.tempimage_type,
                        './tmp/voice_tmp.mp3',
                         os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_add.mp4'))

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
    positions = []
    fps,imagepaths = video_init(opt,path)[:2]

    for i,imagepath in enumerate(imagepaths,1):
        img = impro.imread(os.path.join('./tmp/video2image',imagepath))
        img = runmodel.run_styletransfer(opt, netG, img)
        cv2.imwrite(os.path.join('./tmp/style_transfer',imagepath),img)
        print('\r','Transfer:'+str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print()
    suffix = os.path.basename(opt.model_path).replace('.pth','').replace('style_','')
    ffmpeg.image2video( fps,
                './tmp/style_transfer/output_%05d.'+opt.tempimage_type,
                './tmp/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_'+suffix+'.mp4'))  

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt,netM,imagepaths,savemask=True):
    # get mosaic position
    positions = []
    for i,imagepath in enumerate(imagepaths,1):
        img_origin = impro.imread(os.path.join('./tmp/video2image',imagepath))
        x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
        if savemask:
            cv2.imwrite(os.path.join('./tmp/mosaic_mask',imagepath), mask)
        positions.append([x,y,size])
        print('\r','Find mosaic location:'+str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print('\nOptimize mosaic locations...')
    positions =np.array(positions)
    for i in range(3):positions[:,i] = filt.medfilt(positions[:,i],opt.medfilt_num)
    return positions

def cleanmosaic_img(opt,netG,netM):

    path = opt.media_path
    print('Clean Mosaic:',path)
    img_origin = impro.imread(path)
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    #cv2.imwrite('./mask/'+os.path.basename(path), mask)
    img_result = img_origin.copy()
    if size != 0 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,x,y,size,opt.no_feather)
    else:
        print('Do not find mosaic')
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.jpg'),img_result)

def cleanmosaic_video_byframe(opt,netG,netM):
    path = opt.media_path
    fps,imagepaths = video_init(opt,path)[:2]
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=False)
    # clean mosaic
    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        img_origin = impro.imread(os.path.join('./tmp/video2image',imagepath))
        img_result = img_origin.copy()
        if size != 0:
            img_mosaic = img_origin[y-size:y+size,x-size:x+size]
            if opt.traditional:
                img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
            else:
                img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,x,y,size,opt.no_feather)
        cv2.imwrite(os.path.join('./tmp/replace_mosaic',imagepath),img_result)
        print('\r','Clean Mosaic:'+str(i+1)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print()
    ffmpeg.image2video( fps,
                './tmp/replace_mosaic/output_%05d.'+opt.tempimage_type,
                './tmp/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))  

def cleanmosaic_video_fusion(opt,netG,netM):
    path = opt.media_path
    N = 25
    INPUT_SIZE = 128
    fps,imagepaths,height,width = video_init(opt,path)
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)
    
    # clean mosaic
    img_pool = np.zeros((height,width,3*N), dtype='uint8')
    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        
        # image read stream
        mask = cv2.imread(os.path.join('./tmp/mosaic_mask',imagepath),0)
        if i==0 :
            for j in range(0,N):
                img_pool[:,:,j*3:(j+1)*3] = impro.imread(os.path.join('./tmp/video2image',imagepaths[np.clip(i+j-12,0,len(imagepaths)-1)]))
        else:
            img_pool[:,:,0:(N-1)*3] = img_pool[:,:,3:N*3]
            img_pool[:,:,(N-1)*3:] = impro.imread(os.path.join('./tmp/video2image',imagepaths[np.clip(i+12,0,len(imagepaths)-1)]))
        img_origin = img_pool[:,:,int((N-1)/2)*3:(int((N-1)/2)+1)*3]
        
        if size==0: # can not find mosaic,
            cv2.imwrite(os.path.join('./tmp/replace_mosaic',imagepath),img_origin)
        else:

            mosaic_input = np.zeros((INPUT_SIZE,INPUT_SIZE,3*N+1), dtype='uint8')
            mosaic_input[:,:,0:N*3] = impro.resize(img_pool[y-size:y+size,x-size:x+size,:], INPUT_SIZE)
            mask = impro.resize(mask,np.min(img_origin.shape[:2]))[y-size:y+size,x-size:x+size]
            mosaic_input[:,:,-1] = impro.resize(mask, INPUT_SIZE)

            mosaic_input = data.im2tensor(mosaic_input,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False,is0_1 = False)
            unmosaic_pred = netG(mosaic_input)
            img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = False ,is0_1 = False)
            img_result = impro.replace_mosaic(img_origin,img_fake,x,y,size,opt.no_feather)
            cv2.imwrite(os.path.join('./tmp/replace_mosaic',imagepath),img_result)
        print('\r','Clean Mosaic:'+str(i+1)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),end='')
    print()
    ffmpeg.image2video( fps,
                './tmp/replace_mosaic/output_%05d.'+opt.tempimage_type,
                './tmp/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))        