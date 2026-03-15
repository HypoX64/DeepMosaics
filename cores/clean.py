import os
import time
import numpy as np
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
from models import runmodel
from models.enhancer import enhance_patch
from util import data,util,ffmpeg,filt
from util import image_processing as impro
from .init import video_init
from queue import Queue
from threading import Thread

BATCH_SIZE = 4
WRITE_WORKERS = 4

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt,netM,imagepaths,savemask=True):
    # resume
    continue_flag = False
    if os.path.isfile(os.path.join(opt.temp_dir,'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir,'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step'])>2:
            pre_positions = np.load(os.path.join(opt.temp_dir,'mosaic_positions.npy'))
            return pre_positions
        if int(step['step'])>=2 and resume_frame>0:
            pre_positions = np.load(os.path.join(opt.temp_dir,'mosaic_positions.npy'))
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]
            
    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)
    print('Step:2/4 -- Find mosaic location')

    img_read_pool = Queue(BATCH_SIZE * 3)
    def loader(imagepaths):
        for imagepath in imagepaths:
            img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
            img_read_pool.put(img_origin)
    t = Thread(target=loader,args=(imagepaths,))
    t.setDaemon(True)
    t.start()

    write_executor = ThreadPoolExecutor(max_workers=WRITE_WORKERS) if savemask else None
    total = len(imagepaths)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_imgs = [img_read_pool.get() for _ in range(batch_end - batch_start)]
        batch_masks = runmodel.run_segment_batch(batch_imgs, netM, size=360, gpu_id=opt.gpu_id)

        for j, img_origin in enumerate(batch_imgs):
            idx = batch_start + j
            h, w = img_origin.shape[:2]
            mask = batch_masks[j]
            mask_proc = impro.mask_threshold(mask, ex_mun=int(min(h,w)/20), threshold=opt.mask_threshold)
            if not opt.all_mosaic_area:
                mask_proc = impro.find_mostlikely_ROI(mask_proc)
            x,y,size,area = impro.boundingSquare(mask_proc, Ex_mul=opt.ex_mult)
            rat = min(h,w)/360.0
            x,y,size = int(rat*x),int(rat*y),int(rat*size)
            x,y = np.clip(x, 0, w),np.clip(y, 0, h)
            size = np.clip(size, 0, min(w-x,h-y))
            positions.append([x,y,size])

            if savemask:
                path_out = os.path.join(opt.temp_dir+'/mosaic_mask', imagepaths[idx])
                write_executor.submit(cv2.imwrite, path_out, mask)

            if not opt.no_preview:
                cv2.imshow('mosaic mask', mask)
                cv2.waitKey(1) & 0xFF

        i = batch_end
        if i % 1000 < BATCH_SIZE and i >= BATCH_SIZE:
            save_positions = np.array(positions)
            if continue_flag:
                save_positions = np.concatenate((pre_positions,save_positions),axis=0)
            np.save(os.path.join(opt.temp_dir,'mosaic_positions.npy'),save_positions)
            step = {'step':2,'frame':i+resume_frame}
            util.savejson(os.path.join(opt.temp_dir,'step.json'),step)

        t2 = time.time()
        print('\r',str(i)+'/'+str(total),util.get_bar(100*i/total,num=35),util.counttime(t1,t2,i,total),end='')

    if write_executor:
        write_executor.shutdown(wait=True)
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('\nOptimize mosaic locations...')
    positions =np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions,positions),axis=0)
    for i in range(3):positions[:,i] = filt.medfilt(positions[:,i],opt.medfilt_num)
    step = {'step':3,'frame':0}
    util.savejson(os.path.join(opt.temp_dir,'step.json'),step)
    np.save(os.path.join(opt.temp_dir,'mosaic_positions.npy'),positions)

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
        if getattr(opt, 'enhance', False):
            img_fake = enhance_patch(img_fake, opt.gpu_id)
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
        if getattr(opt, 'enhance', False):
            img_fake = enhance_patch(img_fake, opt.gpu_id)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    return img_result

def cleanmosaic_video_byframe(opt,netG,netM):
    path = opt.media_path
    fps,imagepaths,height,width = video_init(opt,path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)[(start_frame-1):]

    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_exec = ThreadPoolExecutor(max_workers=WRITE_WORKERS)

    img_read_pool = Queue(8)
    def frame_loader(paths):
        for p in paths:
            img_read_pool.put(impro.imread(os.path.join(opt.temp_dir+'/video2image', p)))
    lt = Thread(target=frame_loader, args=(imagepaths,))
    lt.setDaemon(True)
    lt.start()

    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        img_origin = img_read_pool.get()
        img_result = img_origin.copy()
        if size > 100:
            try:
                img_mosaic = img_origin[y-size:y+size,x-size:x+size]
                if opt.traditional:
                    img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
                else:
                    img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
                if getattr(opt, 'enhance', False):
                    img_fake = enhance_patch(img_fake, opt.gpu_id)
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            except Exception as e:
                print('Warning:',e)

        out_path = os.path.join(opt.temp_dir+'/replace_mosaic',imagepath)
        rm_path = os.path.join(opt.temp_dir+'/video2image',imagepath)
        def _write_and_rm(op, rp, img):
            cv2.imwrite(op, img)
            os.remove(rp)
        write_exec.submit(_write_and_rm, out_path, rm_path, img_result)

        if not opt.no_preview:
            cv2.imshow('clean',img_result)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print(flush=True)
    write_exec.shutdown(wait=True, cancel_futures=False)
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video', flush=True)
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
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)[(start_frame-1):]
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_pool = Queue(8)
    show_pool = Queue(4)
    ac_dev = runmodel._get_autocast_device(opt.gpu_id)

    def write_single(save_ori, imagepath, img_origin, img_fake, x, y, size):
        if save_ori:
            img_result = img_origin
        else:
            mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
            img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
        if not opt.no_preview:
            show_pool.put(img_result.copy())
        cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath),img_result)
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))

    write_executor = ThreadPoolExecutor(max_workers=WRITE_WORKERS)

    img_cache = {}
    prefetch_queue = Queue(16)
    _prefetch_error = [None]

    def prefetch_loader():
        try:
            needed = set()
            for i in range(length):
                if i == 0:
                    for j in range(POOL_NUM):
                        needed.add(np.clip(i+j-LEFT_FRAME,0,length-1))
                else:
                    needed.add(np.clip(i+LEFT_FRAME,0,length-1))
            for idx in sorted(needed):
                img = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[idx]))
                prefetch_queue.put((idx, img))
        except Exception as e:
            _prefetch_error[0] = e
        finally:
            prefetch_queue.put(None)

    prefetch_t = Thread(target=prefetch_loader, daemon=True)
    prefetch_t.start()

    def ensure_cached(idx):
        while idx not in img_cache:
            item = prefetch_queue.get(timeout=30)
            if item is None:
                if _prefetch_error[0]:
                    raise RuntimeError('Prefetch failed: %s' % _prefetch_error[0])
                return
            k, v = item
            img_cache[k] = v

    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        input_stream = []
        if i==0:
            for j in range(POOL_NUM):
                frame_idx = np.clip(i+j-LEFT_FRAME,0,length-1)
                ensure_cached(frame_idx)
                img_pool.append(img_cache[frame_idx])
        else:
            img_pool.pop(0)
            frame_idx = np.clip(i+LEFT_FRAME,0,length-1)
            ensure_cached(frame_idx)
            img_pool.append(img_cache[frame_idx])
            # Free frames we no longer need
            old_idx = np.clip(i-LEFT_FRAME-1,0,length-1)
            img_cache.pop(old_idx, None)
        img_origin = img_pool[LEFT_FRAME]

        # preview result and print
        if not opt.no_preview:
            try:
                pool_full = show_pool.qsize() > 3
            except NotImplementedError:
                pool_full = not show_pool.empty()
            if pool_full:
                cv2.imshow('clean',show_pool.get())
                cv2.waitKey(1) & 0xFF

        if size>50:
            try:
                for pos in FRAME_POS:
                    input_stream.append(impro.resize(img_pool[pos][y-size:y+size,x-size:x+size], INPUT_SIZE,interpolation=cv2.INTER_CUBIC)[:,:,::-1])
                if init_flag:
                    init_flag = False
                    previous_frame = input_stream[N]
                    previous_frame = data.im2tensor(previous_frame,bgr2rgb=True,gpu_id=opt.gpu_id)
                
                input_stream = np.array(input_stream).reshape(1,T,INPUT_SIZE,INPUT_SIZE,3).transpose((0,4,1,2,3))
                input_stream = data.to_tensor(data.normalize(input_stream),gpu_id=opt.gpu_id)
                with torch.no_grad():
                    if ac_dev:
                        with torch.autocast(ac_dev, dtype=torch.float16):
                            unmosaic_pred = netG(input_stream,previous_frame)
                    else:
                        unmosaic_pred = netG(input_stream,previous_frame)
                img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = True)
                if getattr(opt, 'enhance', False):
                    img_fake = enhance_patch(img_fake, opt.gpu_id)
                previous_frame = unmosaic_pred
                write_executor.submit(write_single, False, imagepath, img_origin.copy(), img_fake.copy(), x, y, size)
            except Exception as e:
                init_flag = True
                print('Error:',e)
        else:
            write_executor.submit(write_single, True, imagepath, img_origin.copy(), -1, -1, -1, -1)
            init_flag = True
        
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print(flush=True)
    write_executor.shutdown(wait=True, cancel_futures=False)
    while not show_pool.empty():
        show_pool.get_nowait()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video', flush=True)
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4')) 