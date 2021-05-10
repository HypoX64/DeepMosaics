import os
import sys
sys.path.append("..")
from cores import Options
opt = Options()

import random
import datetime
import time

import numpy as np
import cv2
import torch

from models import runmodel,loadmodel
import util.image_processing as impro
from util import filt, util,mosaic,data,ffmpeg


opt.parser.add_argument('--datadir',type=str,default='your video dir', help='')
opt.parser.add_argument('--savedir',type=str,default='../datasets/video/face', help='')
opt.parser.add_argument('--interval',type=int,default=30, help='interval of split video ')
opt.parser.add_argument('--time',type=int,default=5, help='split video time')
opt.parser.add_argument('--minmaskarea',type=int,default=2000, help='')
opt.parser.add_argument('--quality', type=int ,default= 45,help='minimal quality')
opt.parser.add_argument('--outsize', type=int ,default= 286,help='')
opt.parser.add_argument('--startcnt', type=int ,default= 0,help='')
opt.parser.add_argument('--minsize', type=int ,default= 96,help='minimal roi size')
opt.parser.add_argument('--no_sclectscene', action='store_true', help='')  
opt = opt.getparse()


util.makedirs(opt.savedir)
util.writelog(os.path.join(opt.savedir,'opt.txt'), 
              str(time.asctime(time.localtime(time.time())))+'\n'+util.opt2str(opt))

videopaths = util.Traversal(opt.datadir)
videopaths = util.is_videos(videopaths)
random.shuffle(videopaths)

# def network
net = loadmodel.bisenet(opt,'roi')

result_cnt = opt.startcnt
video_cnt = 1
starttime = datetime.datetime.now()
for videopath in videopaths:
    try:
        if opt.no_sclectscene:
            timestamps=['00:00:00']
        else:
            timestamps=[]
            fps,endtime,height,width = ffmpeg.get_video_infos(videopath)
            for cut_point in range(1,int((endtime-opt.time)/opt.interval)):
                util.clean_tempfiles(opt)
                ffmpeg.video2image(videopath, opt.temp_dir+'/video2image/%05d.'+opt.tempimage_type,fps=1,
                    start_time = util.second2stamp(cut_point*opt.interval),last_time = util.second2stamp(opt.time))
                imagepaths = util.Traversal(opt.temp_dir+'/video2image')
                imagepaths = sorted(imagepaths)
                cnt = 0 
                for i in range(opt.time):
                    img = impro.imread(imagepaths[i])
                    mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
                    if not opt.all_mosaic_area:
                        mask = impro.find_mostlikely_ROI(mask)
                    x,y,size,area = impro.boundingSquare(mask,Ex_mul=1)
                    if area > opt.minmaskarea and size>opt.minsize and impro.Q_lapulase(img)>opt.quality:
                        cnt +=1
                if cnt == opt.time:
                    # print(second)
                    timestamps.append(util.second2stamp(cut_point*opt.interval))
        util.writelog(os.path.join(opt.savedir,'opt.txt'),videopath+'\n'+str(timestamps))
        #print(timestamps)

        #generate datasets
        print('Generate datasets...')
        for timestamp in timestamps:
            savecnt = '%05d' % result_cnt
            origindir = os.path.join(opt.savedir,savecnt,'origin_image')
            maskdir = os.path.join(opt.savedir,savecnt,'mask')
            util.makedirs(origindir)
            util.makedirs(maskdir)

            util.clean_tempfiles(opt)
            ffmpeg.video2image(videopath, opt.temp_dir+'/video2image/%05d.'+opt.tempimage_type,
                start_time = timestamp,last_time = util.second2stamp(opt.time))
            
            endtime = datetime.datetime.now()
            print(str(video_cnt)+'/'+str(len(videopaths))+' ',
                util.get_bar(100*video_cnt/len(videopaths),35),'',
                util.second2stamp((endtime-starttime).seconds)+'/'+util.second2stamp((endtime-starttime).seconds/video_cnt*len(videopaths)))

            imagepaths = util.Traversal(opt.temp_dir+'/video2image')
            imagepaths = sorted(imagepaths)
            imgs=[];masks=[]
            # mask_flag = False
            # for imagepath in imagepaths:
            #     img = impro.imread(imagepath)
            #     mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
            #     imgs.append(img)
            #     masks.append(mask)
            #     if not mask_flag:
            #         mask_avg = mask.astype(np.float64)
            #         mask_flag = True
            #     else:
            #         mask_avg += mask.astype(np.float64)

            # mask_avg = np.clip(mask_avg/len(imagepaths),0,255).astype('uint8')
            # mask_avg = impro.mask_threshold(mask_avg,20,64)
            # if not opt.all_mosaic_area:
            #     mask_avg = impro.find_mostlikely_ROI(mask_avg)
            # x,y,size,area = impro.boundingSquare(mask_avg,Ex_mul=random.uniform(1.1,1.5))
            
            # for i in range(len(imagepaths)):
            #     img = impro.resize(imgs[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC) 
            #     mask = impro.resize(masks[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
            #     impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)
            ex_mul = random.uniform(1.2,1.7)
            positions = []
            for imagepath in imagepaths:
                img = impro.imread(imagepath)
                mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
                imgs.append(img)
                masks.append(mask)
                x,y,size,area = impro.boundingSquare(mask,Ex_mul=ex_mul)
                positions.append([x,y,size])
            positions =np.array(positions)
            for i in range(3):positions[:,i] = filt.medfilt(positions[:,i],opt.medfilt_num)

            for i,imagepath in enumerate(imagepaths):
                x,y,size = positions[i][0],positions[i][1],positions[i][2]
                tmp_cnt = i
                while size<opt.minsize//2:
                    tmp_cnt = tmp_cnt-1
                    x,y,size = positions[tmp_cnt][0],positions[tmp_cnt][1],positions[tmp_cnt][2]
                img = impro.resize(imgs[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
                mask = impro.resize(masks[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
                impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
                impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)
                # x_tmp,y_tmp,size_tmp

            # for imagepath in imagepaths:
            #     img = impro.imread(imagepath)
            #     mask,x,y,halfsize,area = runmodel.get_ROI_position(img,net,opt,keepsize=True)
            #     if halfsize>opt.minsize//4:
            #         if not opt.all_mosaic_area:
            #             mask_avg = impro.find_mostlikely_ROI(mask_avg)
            #         x,y,size,area = impro.boundingSquare(mask_avg,Ex_mul=ex_mul)
            #     img = impro.resize(imgs[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     mask = impro.resize(masks[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
            #     impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)


            result_cnt+=1

    except Exception as e:
        video_cnt +=1
        util.writelog(os.path.join(opt.savedir,'opt.txt'), 
              videopath+'\n'+str(result_cnt)+'\n'+str(e))
    video_cnt +=1
    if opt.gpu_id != '-1':
        torch.cuda.empty_cache()
