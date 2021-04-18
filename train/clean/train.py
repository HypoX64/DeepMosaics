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
from multiprocessing import Process, Queue

from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro
from models import pix2pix_model,pix2pixHD_model,video_model,unet_model,loadmodel,videoHD_model,BVDNet,model_util
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

'''
--------------------------Get options--------------------------
'''
opt.parser.add_argument('--N',type=int,default=2, help='The input tensor shape is H×W×T×C, T = 2N+1')
opt.parser.add_argument('--S',type=int,default=3, help='Stride of 3 frames')
# opt.parser.add_argument('--T',type=int,default=7, help='T = 2N+1')
opt.parser.add_argument('--M',type=int,default=100, help='How many frames read from each videos')
opt.parser.add_argument('--lr',type=float,default=0.001, help='')
opt.parser.add_argument('--beta1',type=float,default=0.9, help='')
opt.parser.add_argument('--beta2',type=float,default=0.999, help='')
opt.parser.add_argument('--finesize',type=int,default=256, help='')
opt.parser.add_argument('--loadsize',type=int,default=286, help='')
opt.parser.add_argument('--batchsize',type=int,default=1, help='')
opt.parser.add_argument('--lambda_VGG',type=float,default=0.1, help='')
opt.parser.add_argument('--load_thread',type=int,default=4, help='number of thread for loading data')

opt.parser.add_argument('--dataset',type=str,default='./datasets/face/', help='')
opt.parser.add_argument('--dataset_test',type=str,default='./datasets/face_test/', help='')
opt.parser.add_argument('--n_epoch',type=int,default=200, help='')
opt.parser.add_argument('--save_freq',type=int,default=100000, help='')
opt.parser.add_argument('--continue_train', action='store_true', help='')
opt.parser.add_argument('--savename',type=str,default='face', help='')
opt.parser.add_argument('--showresult_freq',type=int,default=1000, help='')
opt.parser.add_argument('--showresult_num',type=int,default=4, help='')
opt.parser.add_argument('--psnr_freq',type=int,default=100, help='')

class TrainVideoLoader(object):
    """docstring for VideoLoader
    Load a single video(Converted to images)
    How to use:
    1.Init TrainVideoLoader as loader
    2.Get data by loader.ori_stream
    3.loader.next()  to get next stream
    """
    def __init__(self, opt, video_dir, test_flag=False):
        super(TrainVideoLoader, self).__init__()
        self.opt = opt
        self.test_flag = test_flag
        self.video_dir = video_dir
        self.t = 0
        self.n_iter = self.opt.M -self.opt.S*(self.opt.T+1)
        self.transform_params = data.get_transform_params()
        self.ori_load_pool = []
        self.mosaic_load_pool = []
        self.previous_pred = None
        feg_ori =  impro.imread(os.path.join(video_dir,'origin_image','00001.jpg'),loadsize=self.opt.loadsize,rgb=True)
        feg_mask = impro.imread(os.path.join(video_dir,'mask','00001.png'),mod='gray',loadsize=self.opt.loadsize)
        self.mosaic_size,self.mod,self.rect_rat,self.feather = mosaic.get_random_parameter(feg_ori,feg_mask)
        self.startpos = [random.randint(0,self.mosaic_size),random.randint(0,self.mosaic_size)]

        #Init load pool
        for i in range(self.opt.S*self.opt.T):
            #print(os.path.join(video_dir,'origin_image','%05d' % (i+1)+'.jpg'))
            _ori_img = impro.imread(os.path.join(video_dir,'origin_image','%05d' % (i+1)+'.jpg'),loadsize=self.opt.loadsize,rgb=True)
            _mask = impro.imread(os.path.join(video_dir,'mask','%05d' % (i+1)+'.png' ),mod='gray',loadsize=self.opt.loadsize)
            _mosaic_img = mosaic.addmosaic_base(_ori_img, _mask, self.mosaic_size,0, self.mod,self.rect_rat,self.feather,self.startpos)
            self.ori_load_pool.append(self.normalize(_ori_img))
            self.mosaic_load_pool.append(self.normalize(_mosaic_img))
        self.ori_load_pool = np.array(self.ori_load_pool)
        self.mosaic_load_pool = np.array(self.mosaic_load_pool)

        #Init frist stream
        self.ori_stream    = self.ori_load_pool   [np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()
        self.mosaic_stream = self.mosaic_load_pool[np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()
        # stream B,T,H,W,C -> B,C,T,H,W
        self.ori_stream    = self.ori_stream.reshape   (1,self.opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
        self.mosaic_stream = self.mosaic_stream.reshape(1,self.opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
        
        #Init frist previous frame
        self.previous_pred = self.ori_load_pool[self.opt.S*self.opt.N-1].copy()
        # previous B,C,H,W
        self.previous_pred = self.previous_pred.reshape(1,opt.finesize,opt.finesize,3).transpose((0,3,1,2))
    
    def normalize(self,data):
        '''
        normalize to -1 ~ 1
        '''
        return (data.astype(np.float32)/255.0-0.5)/0.5

    def anti_normalize(self,data):
        return np.clip((data*0.5+0.5)*255,0,255).astype(np.uint8)
    
    def next(self):
        if self.t != 0:
            self.previous_pred = None
            self.ori_load_pool   [:self.opt.S*self.opt.T-1] = self.ori_load_pool   [1:self.opt.S*self.opt.T]
            self.mosaic_load_pool[:self.opt.S*self.opt.T-1] = self.mosaic_load_pool[1:self.opt.S*self.opt.T]
            #print(os.path.join(self.video_dir,'origin_image','%05d' % (self.opt.S*self.opt.T+self.t)+'.jpg'))
            _ori_img = impro.imread(os.path.join(self.video_dir,'origin_image','%05d' % (self.opt.S*self.opt.T+self.t)+'.jpg'),loadsize=self.opt.loadsize,rgb=True)
            _mask = impro.imread(os.path.join(self.video_dir,'mask','%05d' % (self.opt.S*self.opt.T+self.t)+'.png' ),mod='gray',loadsize=self.opt.loadsize)
            _mosaic_img = mosaic.addmosaic_base(_ori_img, _mask, self.mosaic_size,0, self.mod,self.rect_rat,self.feather,self.startpos)

            _ori_img,_mosaic_img = self.normalize(_ori_img),self.normalize(_mosaic_img)
            self.ori_load_pool   [self.opt.S*self.opt.T-1] = _ori_img
            self.mosaic_load_pool[self.opt.S*self.opt.T-1] = _mosaic_img

            self.ori_stream    = self.ori_load_pool   [np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()
            self.mosaic_stream = self.mosaic_load_pool[np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()

            # stream B,T,H,W,C -> B,C,T,H,W
            self.ori_stream    = self.ori_stream.reshape   (1,self.opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
            self.mosaic_stream = self.mosaic_stream.reshape(1,self.opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))

        self.t += 1

class DataLoader(object):
    """DataLoader"""
    def __init__(self, opt, videolist, test_flag=False):
        super(DataLoader, self).__init__()
        self.videolist = []
        self.opt = opt
        self.test_flag = test_flag
        for i in range(self.opt.n_epoch):
            self.videolist += videolist
        random.shuffle(self.videolist)
        self.each_video_n_iter = self.opt.M -self.opt.S*(self.opt.T+1)
        self.n_iter = len(self.videolist)//self.opt.load_thread//self.opt.batchsize*self.each_video_n_iter*self.opt.load_thread
        self.queue = Queue(self.opt.load_thread)
        self.ori_stream = np.zeros((self.opt.batchsize,3,self.opt.T,self.opt.finesize,self.opt.finesize),dtype=np.float32)# B,C,T,H,W
        self.mosaic_stream = np.zeros((self.opt.batchsize,3,self.opt.T,self.opt.finesize,self.opt.finesize),dtype=np.float32)# B,C,T,H,W
        self.previous_pred = np.zeros((self.opt.batchsize,3,self.opt.finesize,self.opt.finesize),dtype=np.float32)
        self.load_init()

    def load(self,videolist):
        for load_video_iter in range(len(videolist)//self.opt.batchsize):
            iter_videolist = videolist[load_video_iter*self.opt.batchsize:(load_video_iter+1)*self.opt.batchsize]
            videoloaders = [TrainVideoLoader(self.opt,os.path.join(self.opt.dataset,iter_videolist[i]),self.test_flag) for i in range(self.opt.batchsize)]
            for each_video_iter in range(self.each_video_n_iter):
                for i in range(self.opt.batchsize):
                    self.ori_stream[i] = videoloaders[i].ori_stream
                    self.mosaic_stream[i] = videoloaders[i].mosaic_stream
                    if each_video_iter == 0:
                        self.previous_pred[i] = videoloaders[i].previous_pred
                    videoloaders[i].next()
                if each_video_iter == 0:
                    self.queue.put([self.ori_stream.copy(),self.mosaic_stream.copy(),self.previous_pred])
                else:
                    self.queue.put([self.ori_stream.copy(),self.mosaic_stream.copy(),None])
    
    def load_init(self):
        ptvn = len(self.videolist)//self.opt.load_thread #pre_thread_video_num
        for i in range(self.opt.load_thread):
            p = Process(target=self.load,args=(self.videolist[i*ptvn:(i+1)*ptvn],))
            p.daemon = True
            p.start()

    def get_data(self):
        return self.queue.get()

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
net = BVDNet.BVDNet(opt.N)


if opt.use_gpu != '-1' and len(opt.use_gpu) == 1:
    torch.backends.cudnn.benchmark = True
    net.cuda()
elif opt.use_gpu != '-1' and len(opt.use_gpu) > 1:
    torch.backends.cudnn.benchmark = True
    net = nn.DataParallel(net)
    net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
lossf_L1 = nn.L1Loss()
lossf_VGG = BVDNet.VGGLoss([opt.use_gpu])

videolist_tmp = os.listdir(opt.dataset)
videolist = []
for video in videolist_tmp:
    if os.path.isdir(os.path.join(opt.dataset,video)):
        if len(os.listdir(os.path.join(opt.dataset,video,'mask')))>=opt.M:
            videolist.append(video)
sorted(videolist)
videolist_train = videolist[:int(len(videolist)*0.8)].copy()
videolist_eval = videolist[int(len(videolist)*0.8):].copy()

dataloader_train = DataLoader(opt, videolist_train)
dataloader_eval = DataLoader(opt, videolist_eval)

previous_predframe_tmp = 0
for train_iter in range(dataloader_train.n_iter):
    t_start = time.time()
    # train
    ori_stream,mosaic_stream,previous_frame = dataloader_train.get_data()
    ori_stream = data.to_tensor(ori_stream, opt.use_gpu)
    mosaic_stream = data.to_tensor(mosaic_stream, opt.use_gpu)
    if previous_frame is None:
        previous_frame = data.to_tensor(previous_predframe_tmp, opt.use_gpu)
    else:
        previous_frame = data.to_tensor(previous_frame, opt.use_gpu)
    optimizer.zero_grad()
    out = net(mosaic_stream,previous_frame)
    loss_L1 = lossf_L1(out,ori_stream[:,:,opt.N])
    loss_VGG = lossf_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
    TBGlobalWriter.add_scalars('loss/train', {'L1':loss_L1.item(),'VGG':loss_VGG.item()}, train_iter)
    loss = loss_L1+loss_VGG
    loss.backward()
    optimizer.step()
    previous_predframe_tmp = out.detach().cpu().numpy()

    # save network
    if train_iter%opt.save_freq == 0 and train_iter != 0:
        model_util.save(net, os.path.join('checkpoints',opt.savename,str(train_iter)+'.pth'), opt.use_gpu)

    # psnr
    if train_iter%opt.psnr_freq ==0:
        psnr = 0
        for i in range(len(out)):
            psnr += impro.psnr(data.tensor2im(out,batch_index=i), data.tensor2im(ori_stream[:,:,opt.N],batch_index=i))
        TBGlobalWriter.add_scalars('psnr', {'train':psnr/len(out)}, train_iter)

    if train_iter % opt.showresult_freq == 0:
        show_imgs = []
        for i in range(opt.showresult_num):
            show_imgs += [data.tensor2im(mosaic_stream[:,:,opt.N],rgb2bgr = False,batch_index=i),
                data.tensor2im(out,rgb2bgr = False,batch_index=i),
                data.tensor2im(ori_stream[:,:,opt.N],rgb2bgr = False,batch_index=i)]
        show_img = impro.splice(show_imgs,  (opt.showresult_num,3))
        TBGlobalWriter.add_image('train', show_img,train_iter,dataformats='HWC')

    # eval
    if (train_iter)%5 ==0:
        ori_stream,mosaic_stream,previous_frame = dataloader_eval.get_data()
        ori_stream = data.to_tensor(ori_stream, opt.use_gpu)
        mosaic_stream = data.to_tensor(mosaic_stream, opt.use_gpu)
        if previous_frame is None:
            previous_frame = data.to_tensor(previous_predframe_tmp, opt.use_gpu)
        else:
            previous_frame = data.to_tensor(previous_frame, opt.use_gpu)
        with torch.no_grad():
            out = net(mosaic_stream,previous_frame)
            loss_L1 = lossf_L1(out,ori_stream[:,:,opt.N])
            loss_VGG = lossf_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
        TBGlobalWriter.add_scalars('loss/eval', {'L1':loss_L1.item(),'VGG':loss_VGG.item()}, train_iter)
        previous_predframe_tmp = out.detach().cpu().numpy()

        #psnr
        if (train_iter)%opt.psnr_freq ==0:
            psnr = 0
            for i in range(len(out)):
                psnr += impro.psnr(data.tensor2im(out,batch_index=i), data.tensor2im(ori_stream[:,:,opt.N],batch_index=i))
            TBGlobalWriter.add_scalars('psnr', {'eval':psnr/len(out)}, train_iter)

        if train_iter % opt.showresult_freq == 0:
            show_imgs = []
            for i in range(opt.showresult_num):
                show_imgs += [data.tensor2im(mosaic_stream[:,:,opt.N],rgb2bgr = False,batch_index=i),
                    data.tensor2im(out,rgb2bgr = False,batch_index=i),
                    data.tensor2im(ori_stream[:,:,opt.N],rgb2bgr = False,batch_index=i)]
            show_img = impro.splice(show_imgs, (opt.showresult_num,3))
            TBGlobalWriter.add_image('eval', show_img,train_iter,dataformats='HWC')
            t_end = time.time()
            print('iter:{0:d}  t:{1:.2f}  l1:{2:.4f}  vgg:{3:.4f}  psnr:{4:.2f}'.format(train_iter,t_end-t_start,
                loss_L1.item(),loss_VGG.item(),psnr/len(out)) )
            t_strat = time.time()

    # test
    if train_iter % opt.showresult_freq == 0 and os.path.isdir(opt.dataset_test):
        show_imgs = []
        videos = os.listdir(opt.dataset_test)
        sorted(videos)
        for video in videos:
            frames = os.listdir(os.path.join(opt.dataset_test,video,'image'))
            sorted(frames)
            mosaic_stream = []
            for i in range(opt.T):
                _mosaic = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[i*opt.S]),loadsize=opt.finesize,rgb=True)
                mosaic_stream.append(_mosaic)
            previous = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[opt.N*opt.S-1]),loadsize=opt.finesize,rgb=True)
            mosaic_stream = (np.array(mosaic_stream).astype(np.float32)/255.0-0.5)/0.5
            mosaic_stream = mosaic_stream.reshape(1,opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
            mosaic_stream = data.to_tensor(mosaic_stream, opt.use_gpu)
            previous = data.im2tensor(previous,bgr2rgb = False, use_gpu = opt.use_gpu,use_transform = False, is0_1 = False)
            with torch.no_grad():
                out = net(mosaic_stream,previous)
            show_imgs+= [data.tensor2im(mosaic_stream[:,:,opt.N],rgb2bgr = False),data.tensor2im(out,rgb2bgr = False)]

        show_img = impro.splice(show_imgs, (len(videos),2))
        TBGlobalWriter.add_image('test', show_img,train_iter,dataformats='HWC')
