import os
import time
import torch
import numpy as np
import cv2

from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro

'''
---------------------Video Init---------------------
'''
def video_init(opt,path):
    fps,endtime,height,width = ffmpeg.get_video_infos(path)
    if opt.fps !=0:
        fps = opt.fps

    continue_flag = False
    imagepaths = []

    if os.path.isdir(opt.temp_dir):
        imagepaths = os.listdir(opt.temp_dir+'/video2image')
        if imagepaths != []:
            imagepaths.sort()
            last_frame = int(imagepaths[-1][7:13])
            if (opt.last_time != '00:00:00' and  last_frame > fps*(util.stamp2second(opt.last_time)-1)) \
            or (opt.last_time == '00:00:00' and last_frame > fps*(endtime-1)):            
                choose = input('There is an unfinished video. Continue it? [y/n] ')
                if choose.lower() =='yes' or choose.lower() == 'y':
                    continue_flag = True
    
    if not continue_flag:
        print('Step:1/4 -- Convert video to images')
        util.file_init(opt)
        ffmpeg.video2voice(path,opt.temp_dir+'/voice_tmp.mp3',opt.start_time,opt.last_time)
        ffmpeg.video2image(path,opt.temp_dir+'/video2image/output_%06d.'+opt.tempimage_type,fps,opt.start_time,opt.last_time)
        imagepaths = os.listdir(opt.temp_dir+'/video2image')
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
    length = len(imagepaths)
    # get position
    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
    
    print('Step:2/4 -- Find ROI location')
    for i,imagepath in enumerate(imagepaths,1):
        img = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        mask,x,y,size,area = runmodel.get_ROI_position(img,netS,opt)
        positions.append([x,y,area])      
        cv2.imwrite(os.path.join(opt.temp_dir+'/ROI_mask',imagepath),mask)     
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('preview',mask)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i,length),end='')
    
    print('\nOptimize ROI locations...')
    mask_index = filt.position_medfilt(np.array(positions), 7)

    # add mosaic
    print('Step:3/4 -- Add Mosaic:')
    t1 = time.time()
    for i,imagepath in enumerate(imagepaths,1):
        mask = impro.imread(os.path.join(opt.temp_dir+'/ROI_mask',imagepaths[mask_index[i-1]]),'gray')
        img = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        if impro.mask_area(mask)>100: 
            try:#Avoid unknown errors
                img = mosaic.addmosaic(img, mask, opt)
            except Exception as e:
                   print('Warning:',e)
        cv2.imwrite(os.path.join(opt.temp_dir+'/addmosaic_image',imagepath),img)
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('preview',img)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i,length),end='')
    
    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                        opt.temp_dir+'/addmosaic_image/output_%06d.'+opt.tempimage_type,
                        opt.temp_dir+'/voice_tmp.mp3',
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

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt,netM,imagepaths,savemask=True):
    # get mosaic position
    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)
    print('Step:2/4 -- Find mosaic location')
    for i,imagepath in enumerate(imagepaths,1):
        img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
        positions.append([x,y,size])
        if savemask:
            cv2.imwrite(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath), mask)
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('mosaic mask',mask)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),util.counttime(t1,t2,i,len(imagepaths)),end='')
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
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
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    else:
        print('Do not find mosaic')
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.jpg'),img_result)

def cleanmosaic_img_server(opt,img_origin,netG,netM):
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    img_result = img_origin.copy()
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    return img_result

def cleanmosaic_video_byframe(opt,netG,netM):
    path = opt.media_path
    fps,imagepaths = video_init(opt,path)[:2]
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        img_result = img_origin.copy()
        if size > 100:
            try:#Avoid unknown errors
                img_mosaic = img_origin[y-size:y+size,x-size:x+size]
                if opt.traditional:
                    img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
                else:
                    img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            except Exception as e:
                print('Warning:',e)
        cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath),img_result)
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('clean',img_result)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))  

def cleanmosaic_video_fusion(opt,netG,netM):
    path = opt.media_path
    N,T,S = 2,5,3
    LEFT_FRAME = (N*S)
    POOL_NUM = LEFT_FRAME*2+1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S,T,dtype=np.int64)
    img_pool = []
    previous_frame = None
    init_flag = True
    
    fps,imagepaths,height,width = video_init(opt,path)
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    
    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        input_stream = []
        # image read stream
        if i==0 :# init
            for j in range(POOL_NUM):
                img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+j-LEFT_FRAME,0,len(imagepaths)-1)])))
        else: # load next frame
            img_pool.pop(0)
            img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+LEFT_FRAME,0,len(imagepaths)-1)])))
        img_origin = img_pool[LEFT_FRAME]
        img_result = img_origin.copy()

        if size>50:
            try:#Avoid unknown errors
                for pos in FRAME_POS:
                    input_stream.append(impro.resize(img_pool[pos][y-size:y+size,x-size:x+size], INPUT_SIZE)[:,:,::-1])
                if init_flag:
                    init_flag = False
                    previous_frame = input_stream[N]
                    previous_frame = data.im2tensor(previous_frame,bgr2rgb=True,gpu_id=opt.gpu_id)
                
                input_stream = np.array(input_stream).reshape(1,T,INPUT_SIZE,INPUT_SIZE,3).transpose((0,4,1,2,3))
                input_stream = data.to_tensor(data.normalize(input_stream),gpu_id=opt.gpu_id)
                with torch.no_grad():
                    unmosaic_pred = netG(input_stream,previous_frame)
                img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = True)
                previous_frame = unmosaic_pred
                # previous_frame = data.tensor2im(unmosaic_pred,rgb2bgr = True)
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            except Exception as e:
                init_flag = True
                print('Error:',e)
        else:
            init_flag = True
        cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath),img_result)
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))      
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('clean',img_result)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))        