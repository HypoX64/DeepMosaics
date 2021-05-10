import os
import sys
sys.path.append("..")
from cores import Options
opt = Options()

import random
import datetime
import time
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import cv2
import torch

from models import runmodel,loadmodel
import util.image_processing as impro
from util import degradater, util,mosaic,data


opt.parser.add_argument('--datadir',type=str,default='../datasets/draw/face', help='')
opt.parser.add_argument('--savedir',type=str,default='../datasets/pix2pix/face', help='')
opt.parser.add_argument('--name',type=str,default='', help='save name')
opt.parser.add_argument('--mod',type=str,default='drawn', help='drawn | network | irregular | drawn,irregular | network,irregular')
opt.parser.add_argument('--square', action='store_true', help='if specified, crop to square')
opt.parser.add_argument('--irrholedir',type=str,default='../datasets/Irregular_Holes_mask', help='')  
opt.parser.add_argument('--hd', action='store_true', help='if false make dataset for pix2pix, if Ture for pix2pix_HD')
opt.parser.add_argument('--savemask', action='store_true', help='if specified,save mask')
opt.parser.add_argument('--outsize', type=int ,default= 512,help='')
opt.parser.add_argument('--fold', type=int ,default= 1,help='')
opt.parser.add_argument('--start', type=int ,default= 0,help='')
opt.parser.add_argument('--minsize', type=int ,default= 128,help='when [square], minimal roi size')
opt.parser.add_argument('--quality', type=int ,default= 40,help='when [square], minimal quality')

opt = opt.getparse()

util.makedirs(opt.savedir)
util.writelog(os.path.join(opt.savedir,'opt.txt'), 
              str(time.asctime(time.localtime(time.time())))+'\n'+util.opt2str(opt))
opt.mod = (opt.mod).split(',')

#save dir
if opt.hd:
    train_A_path = os.path.join(opt.savedir,'train_A')
    train_B_path = os.path.join(opt.savedir,'train_B')
    util.makedirs(train_A_path)
    util.makedirs(train_B_path)
else:
    train_path = os.path.join(opt.savedir,'train')
    util.makedirs(train_path)
if opt.savemask:
    mask_save_path = os.path.join(opt.savedir,'mask')
    util.makedirs(mask_save_path)

#read dir
if 'drawn' in opt.mod:
    imgpaths = util.Traversal(os.path.join(opt.datadir,'origin_image'))
    imgpaths.sort()
    maskpaths = util.Traversal(os.path.join(opt.datadir,'mask'))
    maskpaths.sort()
if 'network' in opt.mod or 'irregular' in opt.mod:
    imgpaths = util.Traversal(opt.datadir)
    imgpaths = util.is_imgs(imgpaths)
    random.shuffle (imgpaths)
if 'irregular' in opt.mod:
    irrpaths = util.Traversal(opt.irrholedir)


#def network                
if 'network' in opt.mod:
    net = loadmodel.bisenet(opt,'roi')

print('Find images:',len(imgpaths))
starttime = datetime.datetime.now()
filecnt = 0
savecnt = opt.start
for fold in range(opt.fold):
    for i in range(len(imgpaths)):
        filecnt += 1
        try:
            # load image and get mask
            img = impro.imread(imgpaths[i])
            if 'drawn' in opt.mod:
                mask_drawn = impro.imread(maskpaths[i],'gray')
                mask_drawn = impro.resize_like(mask_drawn, img)
                mask = mask_drawn
            if 'irregular' in opt.mod:
                mask_irr = impro.imread(irrpaths[random.randint(0,12000-1)],'gray')
                mask_irr = data.random_transform_single_mask(mask_irr, (img.shape[0],img.shape[1]))
                mask = mask_irr
            if 'network' in opt.mod:
                mask_net = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
                if opt.gpu_id != -1:
                    torch.cuda.empty_cache()
                if not opt.all_mosaic_area:
                    mask_net = impro.find_mostlikely_ROI(mask_net)
                mask = mask_net
            if opt.mod == ['drawn','irregular']:
                mask = cv2.bitwise_and(mask_irr, mask_drawn)
            if opt.mod == ['network','irregular']:
                mask = cv2.bitwise_and(mask_irr, mask_net)

                #checkandsave
                # t=threading.Thread(target=checksaveimage,args=(opt,img,mask,))
                # t.start()

                saveflag = True
                if opt.mod == ['drawn','irregular']:
                    x,y,size,area = impro.boundingSquare(mask_drawn, random.uniform(1.1,1.6))
                elif opt.mod == ['network','irregular']:
                    x,y,size,area = impro.boundingSquare(mask_net, random.uniform(1.1,1.6))
                else:
                    x,y,size,area = impro.boundingSquare(mask, random.uniform(1.1,1.6))

                if area < 1000:
                    saveflag = False
                else:
                    if opt.square:
                        if size < opt.minsize:
                            saveflag = False
                        else:
                            img = impro.resize(img[y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
                            mask =  impro.resize(mask[y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
                            if impro.Q_lapulase(img)<opt.quality:
                                saveflag = False         
                    else:
                        img = impro.resize(img,opt.outsize,interpolation=cv2.INTER_CUBIC)
                
                if saveflag:
                    # add mosaic
                    img_mosaic = mosaic.addmosaic_random(img, mask)
                    # random degradater
                    if random.random()>0.5:
                        degradate_params = degradater.get_random_degenerate_params(mod='weaker_2')
                        img = degradater.degradate(img,degradate_params)
                        img_mosaic = degradater.degradate(img_mosaic,degradate_params)
                    # if random.random()>0.5:
                    #     Q = random.randint(1,15)
                    #     img = impro.dctblur(img,Q)
                    #     img_mosaic = impro.dctblur(img_mosaic,Q)

                    savecnt += 1

                    if opt.hd:
                        cv2.imwrite(os.path.join(train_A_path,opt.name+'%06d' % savecnt+'.jpg'), img_mosaic)
                        cv2.imwrite(os.path.join(train_B_path,opt.name+'%06d' % savecnt+'.jpg'), img)
                    else:
                        merge_img = impro.makedataset(img_mosaic, img)
                        cv2.imwrite(os.path.join(train_path,opt.name+'%06d' % savecnt+'.jpg'), merge_img)
                    if opt.savemask:
                        cv2.imwrite(os.path.join(mask_save_path,opt.name+'%06d' % savecnt+'.png'), mask)
                
                # print("Processing:",imgpaths[i]," ","Remain:",len(imgpaths)*opt.fold-filecnt)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image',img_mosaic)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()   
        except Exception as e:
            print(imgpaths[i],e)
        if filecnt%10==0:
            endtime = datetime.datetime.now()
            # used_time = (endtime-starttime).seconds
            used_time = (endtime-starttime).seconds
            all_length = len(imgpaths)*opt.fold 
            percent = round(100*filecnt/all_length,1)
            all_time = used_time/filecnt*all_length

            print('\r','',str(filecnt)+'/'+str(all_length)+' ',
                util.get_bar(percent,25),'',
                util.second2stamp(used_time)+'/'+util.second2stamp(all_time),
                'f:'+str(savecnt),end= " ")