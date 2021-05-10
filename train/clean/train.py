import os
import sys
sys.path.append("..")
sys.path.append("../..")
from cores import Options
opt = Options()

import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import time

from util import util,data,dataloader
from util import image_processing as impro
from models import BVDNet,model_util
from skimage.metrics import structural_similarity
from tensorboardX import SummaryWriter

'''
--------------------------Get options--------------------------
'''
opt.parser.add_argument('--N',type=int,default=2, help='The input tensor shape is H×W×T×C, T = 2N+1')
opt.parser.add_argument('--S',type=int,default=3, help='Stride of 3 frames')
# opt.parser.add_argument('--T',type=int,default=7, help='T = 2N+1')
opt.parser.add_argument('--M',type=int,default=100, help='How many frames read from each videos')
opt.parser.add_argument('--lr',type=float,default=0.0002, help='')
opt.parser.add_argument('--beta1',type=float,default=0.9, help='')
opt.parser.add_argument('--beta2',type=float,default=0.999, help='')
opt.parser.add_argument('--finesize',type=int,default=256, help='')
opt.parser.add_argument('--loadsize',type=int,default=286, help='')
opt.parser.add_argument('--batchsize',type=int,default=1, help='')
opt.parser.add_argument('--no_gan', action='store_true', help='if specified, do not use gan')
opt.parser.add_argument('--n_blocks',type=int,default=4, help='')
opt.parser.add_argument('--n_layers_D',type=int,default=2, help='')
opt.parser.add_argument('--num_D',type=int,default=3, help='')
opt.parser.add_argument('--lambda_L2',type=float,default=100, help='')
opt.parser.add_argument('--lambda_VGG',type=float,default=1, help='')
opt.parser.add_argument('--lambda_GAN',type=float,default=0.01, help='')
opt.parser.add_argument('--lambda_D',type=float,default=1, help='')
opt.parser.add_argument('--load_thread',type=int,default=16, help='number of thread for loading data')

opt.parser.add_argument('--dataset',type=str,default='./datasets/face/', help='')
opt.parser.add_argument('--dataset_test',type=str,default='./datasets/face_test/', help='')
opt.parser.add_argument('--n_epoch',type=int,default=200, help='')
opt.parser.add_argument('--save_freq',type=int,default=10000, help='')
opt.parser.add_argument('--continue_train', action='store_true', help='')
opt.parser.add_argument('--savename',type=str,default='face', help='')
opt.parser.add_argument('--showresult_freq',type=int,default=1000, help='')
opt.parser.add_argument('--showresult_num',type=int,default=4, help='')

def ImageQualityEvaluation(tensor1,tensor2,showiter,writer,tag):
    batch_len = len(tensor1)
    psnr,ssmi = 0,0
    for i in range(len(tensor1)):
        img1,img2 = data.tensor2im(tensor1,rgb2bgr=False,batch_index=i), data.tensor2im(tensor2,rgb2bgr=False,batch_index=i)
        psnr += impro.psnr(img1,img2)
        ssmi += structural_similarity(img1,img2,multichannel=True)
    writer.add_scalars('quality/psnr', {tag:psnr/batch_len}, showiter)
    writer.add_scalars('quality/ssmi', {tag:ssmi/batch_len}, showiter)
    return psnr/batch_len,ssmi/batch_len

def ShowImage(tensor1,tensor2,tensor3,showiter,max_num,writer,tag):
    show_imgs = []
    for i in range(max_num):
        show_imgs += [  data.tensor2im(tensor1,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor2,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor3,rgb2bgr = False,batch_index=i)]
    show_img = impro.splice(show_imgs,  (opt.showresult_num,3))
    writer.add_image(tag, show_img,showiter,dataformats='HWC')

'''
--------------------------Init--------------------------
'''
opt = opt.getparse()
opt.T = 2*opt.N+1
if opt.showresult_num >opt.batchsize:
    opt.showresult_num = opt.batchsize
dir_checkpoint = os.path.join('checkpoints',opt.savename)
util.makedirs(dir_checkpoint)
# start tensorboard
localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
tensorboard_savedir = os.path.join('checkpoints/tensorboard',localtime+'_'+opt.savename)
TBGlobalWriter = SummaryWriter(tensorboard_savedir)
print('Please run "tensorboard --logdir checkpoints/tensorboardX --host=your_server_ip" and input "'+localtime+'" to filter outputs')

'''
--------------------------Init Network--------------------------
'''
if opt.gpu_id != '-1' and len(opt.gpu_id) == 1:
    torch.backends.cudnn.benchmark = True

netG = BVDNet.define_G(opt.N,opt.n_blocks,gpu_id=opt.gpu_id)
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
lossfun_L2 = nn.MSELoss()
lossfun_VGG = model_util.VGGLoss(opt.gpu_id)
if not opt.no_gan:
    netD = BVDNet.define_D(n_layers_D=opt.n_layers_D,num_D=opt.num_D,gpu_id=opt.gpu_id)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    lossfun_GAND = BVDNet.GANLoss('D')
    lossfun_GANG = BVDNet.GANLoss('G')

'''
--------------------------Init DataLoader--------------------------
'''
videolist_tmp = os.listdir(opt.dataset)
videolist = []
for video in videolist_tmp:
    if os.path.isdir(os.path.join(opt.dataset,video)):
        if len(os.listdir(os.path.join(opt.dataset,video,'mask')))>=opt.M:
            videolist.append(video)
sorted(videolist)
videolist_train = videolist[:int(len(videolist)*0.8)].copy()
videolist_eval = videolist[int(len(videolist)*0.8):].copy()

Videodataloader_train = dataloader.VideoDataLoader(opt, videolist_train)
Videodataloader_eval = dataloader.VideoDataLoader(opt, videolist_eval)

'''
--------------------------Train--------------------------
'''
previous_predframe_tmp = 0
for train_iter in range(Videodataloader_train.n_iter):
    t_start = time.time()
    # train
    ori_stream,mosaic_stream,previous_frame = Videodataloader_train.get_data()
    ori_stream = data.to_tensor(ori_stream, opt.gpu_id)
    mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id)
    if previous_frame is None:
        previous_frame = data.to_tensor(previous_predframe_tmp, opt.gpu_id)
    else:
        previous_frame = data.to_tensor(previous_frame, opt.gpu_id)

    ############### Forward ####################
    # Fake Generator
    out = netG(mosaic_stream,previous_frame)
    # Discriminator
    if not opt.no_gan:
        dis_real = netD(torch.cat((mosaic_stream[:,:,opt.N],ori_stream[:,:,opt.N].detach()),dim=1))
        dis_fake_D = netD(torch.cat((mosaic_stream[:,:,opt.N],out.detach()),dim=1))
        loss_D = lossfun_GAND(dis_fake_D,dis_real) * opt.lambda_GAN * opt.lambda_D
    # Generator
    loss_L2 = lossfun_L2(out,ori_stream[:,:,opt.N]) * opt.lambda_L2
    loss_VGG = lossfun_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
    loss_G = loss_L2+loss_VGG
    if not opt.no_gan:
        dis_fake_G = netD(torch.cat((mosaic_stream[:,:,opt.N],out),dim=1))
        loss_GANG = lossfun_GANG(dis_fake_G) * opt.lambda_GAN
        loss_G = loss_G + loss_GANG

    ############### Backward Pass ####################
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    if not opt.no_gan:
        optimizer_D.zero_grad()
        loss_D.backward()        
        optimizer_D.step()

    previous_predframe_tmp = out.detach().cpu().numpy()
    
    if not opt.no_gan:
        TBGlobalWriter.add_scalars('loss/train', {'L2':loss_L2.item(),'VGG':loss_VGG.item(),
            'loss_D':loss_D.item(),'loss_G':loss_G.item()}, train_iter)
    else:
        TBGlobalWriter.add_scalars('loss/train', {'L2':loss_L2.item(),'VGG':loss_VGG.item()}, train_iter)

    # save network
    if train_iter%opt.save_freq == 0 and train_iter != 0:
        model_util.save(netG, os.path.join('checkpoints',opt.savename,str(train_iter)+'_G.pth'), opt.gpu_id)
        if not opt.no_gan:
            model_util.save(netD, os.path.join('checkpoints',opt.savename,str(train_iter)+'_D.pth'), opt.gpu_id)

    # Image quality evaluation 
    if train_iter%(opt.showresult_freq//10) == 0:
        ImageQualityEvaluation(out,ori_stream[:,:,opt.N],train_iter,TBGlobalWriter,'train')
    
    # Show result
    if train_iter % opt.showresult_freq == 0:
        ShowImage(mosaic_stream[:,:,opt.N],out,ori_stream[:,:,opt.N],train_iter,opt.showresult_num,TBGlobalWriter,'train')
    
    '''
    --------------------------Eval--------------------------
    '''
    if (train_iter)%5 ==0:
        ori_stream,mosaic_stream,previous_frame = Videodataloader_eval.get_data()
        ori_stream = data.to_tensor(ori_stream, opt.gpu_id)
        mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id)
        if previous_frame is None:
            previous_frame = data.to_tensor(previous_predframe_tmp, opt.gpu_id)
        else:
            previous_frame = data.to_tensor(previous_frame, opt.gpu_id)
        with torch.no_grad():
            out = netG(mosaic_stream,previous_frame)
            loss_L2 = lossfun_L2(out,ori_stream[:,:,opt.N]) * opt.lambda_L2
            loss_VGG = lossfun_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
        #TBGlobalWriter.add_scalars('loss/eval', {'L2':loss_L2.item(),'VGG':loss_VGG.item()}, train_iter)
        previous_predframe_tmp = out.detach().cpu().numpy()

        # Image quality evaluation 
        if train_iter%(opt.showresult_freq//10) == 0:
            psnr,ssmi = ImageQualityEvaluation(out,ori_stream[:,:,opt.N],train_iter,TBGlobalWriter,'eval')
        
        # Show result
        if train_iter % opt.showresult_freq == 0:
            ShowImage(mosaic_stream[:,:,opt.N],out,ori_stream[:,:,opt.N],train_iter,opt.showresult_num,TBGlobalWriter,'eval')
            t_end = time.time()
            print('iter:{0:d}  t:{1:.2f}  L2:{2:.4f}  vgg:{3:.4f}  psnr:{4:.2f}  ssmi:{5:.3f}'.format(train_iter,t_end-t_start,
                loss_L2.item(),loss_VGG.item(),psnr,ssmi) )
            t_strat = time.time()

    '''
    --------------------------Test--------------------------
    '''
    if train_iter % opt.showresult_freq == 0 and os.path.isdir(opt.dataset_test):
        show_imgs = []
        videos = os.listdir(opt.dataset_test)
        sorted(videos)
        for video in videos:
            frames = os.listdir(os.path.join(opt.dataset_test,video,'image'))
            sorted(frames)
            for step in range(5):
                mosaic_stream = []
                for i in range(opt.T):
                    _mosaic = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[i*opt.S+step]),loadsize=opt.finesize,rgb=True)
                    mosaic_stream.append(_mosaic)
                if step == 0:
                    previous = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[opt.N*opt.S-1]),loadsize=opt.finesize,rgb=True)
                    previous = data.im2tensor(previous,bgr2rgb = False, gpu_id = opt.gpu_id, is0_1 = False)
                mosaic_stream = (np.array(mosaic_stream).astype(np.float32)/255.0-0.5)/0.5
                mosaic_stream = mosaic_stream.reshape(1,opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
                mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id)
                with torch.no_grad():
                    out = netG(mosaic_stream,previous)
                previous = out
            show_imgs+= [data.tensor2im(mosaic_stream[:,:,opt.N],rgb2bgr = False),data.tensor2im(out,rgb2bgr = False)]

        show_img = impro.splice(show_imgs, (len(videos),2))
        TBGlobalWriter.add_image('test', show_img,train_iter,dataformats='HWC')