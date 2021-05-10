import os
import random
import numpy as np
from multiprocessing import Process, Queue
from . import image_processing as impro
from . import mosaic,data

class VideoLoader(object):
    """docstring for VideoLoader
    Load a single video(Converted to images)
    How to use:
    1.Init VideoLoader as loader
    2.Get data by loader.ori_stream
    3.loader.next()  to get next stream
    """
    def __init__(self, opt, video_dir, test_flag=False):
        super(VideoLoader, self).__init__()
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
        self.loadsize = self.opt.loadsize
        #Init load pool
        for i in range(self.opt.S*self.opt.T):
            _ori_img = impro.imread(os.path.join(video_dir,'origin_image','%05d' % (i+1)+'.jpg'),loadsize=self.loadsize,rgb=True)
            _mask = impro.imread(os.path.join(video_dir,'mask','%05d' % (i+1)+'.png' ),mod='gray',loadsize=self.loadsize)
            _mosaic_img = mosaic.addmosaic_base(_ori_img, _mask, self.mosaic_size,0, self.mod,self.rect_rat,self.feather,self.startpos)
            _ori_img = data.random_transform_single_image(_ori_img,opt.finesize,self.transform_params)
            _mosaic_img = data.random_transform_single_image(_mosaic_img,opt.finesize,self.transform_params)

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
        # random
        if np.random.random()<0.05:
            self.startpos = [random.randint(0,self.mosaic_size),random.randint(0,self.mosaic_size)]
        if np.random.random()<0.02:
            self.transform_params['rate']['crop'] = [np.random.random(),np.random.random()]
        if np.random.random()<0.02:
            self.loadsize = np.random.randint(self.opt.finesize,self.opt.loadsize)
        
        if self.t != 0:
            self.previous_pred = None
            self.ori_load_pool   [:self.opt.S*self.opt.T-1] = self.ori_load_pool   [1:self.opt.S*self.opt.T]
            self.mosaic_load_pool[:self.opt.S*self.opt.T-1] = self.mosaic_load_pool[1:self.opt.S*self.opt.T]
            #print(os.path.join(self.video_dir,'origin_image','%05d' % (self.opt.S*self.opt.T+self.t)+'.jpg'))
            _ori_img = impro.imread(os.path.join(self.video_dir,'origin_image','%05d' % (self.opt.S*self.opt.T+self.t)+'.jpg'),loadsize=self.loadsize,rgb=True)
            _mask = impro.imread(os.path.join(self.video_dir,'mask','%05d' % (self.opt.S*self.opt.T+self.t)+'.png' ),mod='gray',loadsize=self.loadsize)
            _mosaic_img = mosaic.addmosaic_base(_ori_img, _mask, self.mosaic_size,0, self.mod,self.rect_rat,self.feather,self.startpos)
            _ori_img = data.random_transform_single_image(_ori_img,self.opt.finesize,self.transform_params)
            _mosaic_img = data.random_transform_single_image(_mosaic_img,self.opt.finesize,self.transform_params)
            
            _ori_img,_mosaic_img = self.normalize(_ori_img),self.normalize(_mosaic_img)
            self.ori_load_pool   [self.opt.S*self.opt.T-1] = _ori_img
            self.mosaic_load_pool[self.opt.S*self.opt.T-1] = _mosaic_img

            self.ori_stream    = self.ori_load_pool   [np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()
            self.mosaic_stream = self.mosaic_load_pool[np.linspace(0, (self.opt.T-1)*self.opt.S,self.opt.T,dtype=np.int64)].copy()

            # stream B,T,H,W,C -> B,C,T,H,W
            self.ori_stream    = self.ori_stream.reshape   (1,self.opt.T,self.opt.finesize,self.opt.finesize,3).transpose((0,4,1,2,3))
            self.mosaic_stream = self.mosaic_stream.reshape(1,self.opt.T,self.opt.finesize,self.opt.finesize,3).transpose((0,4,1,2,3))

        self.t += 1

class VideoDataLoader(object):
    """VideoDataLoader"""
    def __init__(self, opt, videolist, test_flag=False):
        super(VideoDataLoader, self).__init__()
        self.videolist = []
        self.opt = opt
        self.test_flag = test_flag
        for i in range(self.opt.n_epoch):
            self.videolist += videolist.copy()
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
            videoloaders = [VideoLoader(self.opt,os.path.join(self.opt.dataset,iter_videolist[i]),self.test_flag) for i in range(self.opt.batchsize)]
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