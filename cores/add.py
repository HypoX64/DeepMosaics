import os
from queue import Queue
from threading import Thread
import time
import numpy as np
import cv2
from models import runmodel
from util import mosaic,util,ffmpeg,filt
from util import image_processing as impro
from .init import video_init


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

def get_roi_positions(opt,netS,imagepaths,savemask=True):
    # resume
    continue_flag = False
    if os.path.isfile(os.path.join(opt.temp_dir,'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir,'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step'])>2:
            mask_index = np.load(os.path.join(opt.temp_dir,'mask_index.npy'))
            return mask_index
        if int(step['step'])>=2 and resume_frame>0:
            pre_positions = np.load(os.path.join(opt.temp_dir,'roi_positions.npy'))
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]
            
    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    print('Step:2/4 -- Find mosaic location')

    img_read_pool = Queue(4)
    def loader(imagepaths):
        for imagepath in imagepaths:
            img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
            img_read_pool.put(img_origin)
    t = Thread(target=loader,args=(imagepaths,))
    t.daemon = True
    t.start()

    for i,imagepath in enumerate(imagepaths,1):
        img_origin = img_read_pool.get()
        mask,x,y,size,area = runmodel.get_ROI_position(img_origin,netS,opt)
        positions.append([x,y,area])  
        if savemask:
            t = Thread(target=cv2.imwrite,args=(os.path.join(opt.temp_dir+'/ROI_mask',imagepath), mask,))
            t.start()
        if i%1000==0:
            save_positions = np.array(positions)
            if continue_flag:
                save_positions = np.concatenate((pre_positions,save_positions),axis=0)
            np.save(os.path.join(opt.temp_dir,'roi_positions.npy'),save_positions)
            step = {'step':2,'frame':i+resume_frame}
            util.savejson(os.path.join(opt.temp_dir,'step.json'),step)

        #preview result and print
        if not opt.no_preview:
            cv2.imshow('mask',mask)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),util.counttime(t1,t2,i,len(imagepaths)),end='')
    
    if not opt.no_preview:
        cv2.destroyAllWindows()

    print('\nOptimize ROI locations...')
    if continue_flag:
        positions = np.concatenate((pre_positions,positions),axis=0)
    mask_index = filt.position_medfilt(np.array(positions), 7)
    step = {'step':3,'frame':0}
    util.savejson(os.path.join(opt.temp_dir,'step.json'),step)
    np.save(os.path.join(opt.temp_dir,'roi_positions.npy'),positions)
    np.save(os.path.join(opt.temp_dir,'mask_index.npy'),np.array(mask_index))

    return mask_index

def addmosaic_video(opt,netS):
    path = opt.media_path
    fps,imagepaths = video_init(opt,path)[:2]
    length = len(imagepaths)
    start_frame = int(imagepaths[0][7:13])
    mask_index = get_roi_positions(opt,netS,imagepaths)[(start_frame-1):]

    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

    # add mosaic
    print('Step:3/4 -- Add Mosaic:')
    t1 = time.time()
    # print(mask_index)
    for i,imagepath in enumerate(imagepaths,1):
        mask = impro.imread(os.path.join(opt.temp_dir+'/ROI_mask',imagepaths[np.clip(mask_index[i-1]-start_frame,0,1000000)]),'gray')
        img = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        if impro.mask_area(mask)>100: 
            try:#Avoid unknown errors
                img = mosaic.addmosaic(img, mask, opt)
            except Exception as e:
                   print('Warning:',e)
        t = Thread(target=cv2.imwrite,args=(os.path.join(opt.temp_dir+'/addmosaic_image',imagepath),img))
        t.start()
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