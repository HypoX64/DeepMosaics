import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import time

import sys
sys.path.append("..")
from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro
from cores import Options
from models import pix2pix_model
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn

N = 25
ITER = 1000000
LR = 0.0002
use_gpu = True
CONTINUE = True
# BATCHSIZE = 4
dir_checkpoint = 'checkpoints/'
SAVE_FRE = 5000
start_iter = 0
SIZE = 256
lambda_L1 = 100.0
opt = Options().getparse()
opt.use_gpu=True
videos = os.listdir('./dataset')
videos.sort()
lengths = []
for video in videos:
    video_images = os.listdir('./dataset/'+video+'/ori')
    lengths.append(len(video_images))


netG = pix2pix_model.define_G(3*N+1, 3, 128, 'resnet_9blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
netD = pix2pix_model.define_D(3*2, 64, 'basic', n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[])

if CONTINUE:
    netG.load_state_dict(torch.load(dir_checkpoint+'last_G.pth'))
    netD.load_state_dict(torch.load(dir_checkpoint+'last_D.pth'))
    f = open('./iter','r')
    start_iter = int(f.read())
    f.close()
if use_gpu:
    netG.cuda()
    netD.cuda()
    cudnn.benchmark = True
optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR)
optimizer_D = torch.optim.Adam(netG.parameters(), lr=LR)
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()
criterionGAN = pix2pix_model.GANLoss('lsgan').cuda()

def showresult(img1,img2,img3,name):
    img1 = (img1.cpu().detach().numpy()*255)
    img2 = (img2.cpu().detach().numpy()*255)
    img3 = (img3.cpu().detach().numpy()*255)
    batchsize = img1.shape[0]
    size = img1.shape[3]
    ran =int(batchsize*random.random())
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] =img1[ran].transpose((1, 2, 0))
    showimg[0:size,size:size*2] = img2[ran].transpose((1, 2, 0))
    showimg[0:size,size*2:size*3] = img3[ran].transpose((1, 2, 0))
    cv2.imwrite(name, showimg)


def loaddata():
    video_index = random.randint(0,len(videos)-1)
    video = videos[video_index]
    img_index = random.randint(N,lengths[video_index]- N)
    input_img = np.zeros((SIZE,SIZE,3*N+1), dtype='uint8')
    for i in range(0,N):
        # print('./dataset/'+video+'/mosaic/output_'+'%05d'%(img_index+i-int(N/2))+'.png')
        img = cv2.imread('./dataset/'+video+'/mosaic/output_'+'%05d'%(img_index+i-int(N/2))+'.png')
        img = impro.resize(img,SIZE)
        input_img[:,:,i*3:(i+1)*3] = img
    mask = cv2.imread('./dataset/'+video+'/mask/output_'+'%05d'%(img_index)+'.png',0)
    mask = impro.resize(mask,256)
    mask = impro.mask_threshold(mask,15,128)
    input_img[:,:,-1] = mask
    input_img = data.im2tensor(input_img,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False)

    ground_true = cv2.imread('./dataset/'+video+'/ori/output_'+'%05d'%(img_index)+'.png')
    ground_true = impro.resize(ground_true,SIZE)
    # ground_true = im2tensor(ground_true,use_gpu)
    ground_true = data.im2tensor(ground_true,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False)
    return input_img,ground_true

input_imgs=[]
ground_trues=[]
def preload():
    while 1:
        input_img,ground_true = loaddata()
        input_imgs.append(input_img)
        ground_trues.append(ground_true)
        if len(input_imgs)>10:
            del(input_imgs[0])
            del(ground_trues[0])
import threading
t=threading.Thread(target=preload,args=())  #t为新创建的线程
t.start()
time.sleep(3) #wait frist load


netG.train()
loss_sum = [0.,0.]
loss_plot = [[],[]]
item_plot = []
time_start=time.time()
print("Begin training...")
for iter in range(start_iter+1,ITER):

    # input_img,ground_true = loaddata()
    ran = random.randint(0, 9)
    input_img = input_imgs[ran]
    ground_true = ground_trues[ran]

    pred = netG(input_img)

    fake_AB = torch.cat((input_img[:,int((N+1)/2)*3:(int((N+1)/2)+1)*3,:,:], pred), 1)
    pred_fake = netD(fake_AB.detach())
    loss_D_fake = criterionGAN(pred_fake, False)

    real_AB = torch.cat((input_img[:,int((N+1)/2)*3:(int((N+1)/2)+1)*3,:,:], ground_true), 1)
    pred_real = netD(real_AB)
    loss_D_real = criterionGAN(pred_real, True)
    loss_D = (loss_D_fake + loss_D_real) * 0.5

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    fake_AB = torch.cat((input_img[:,int((N+1)/2)*3:(int((N+1)/2)+1)*3,:,:], pred), 1)
    pred_fake = netD(fake_AB)
    loss_G_GAN = criterionGAN(pred_fake, True)
    # Second, G(A) = B
    loss_G_L1 = criterion_L1(pred, ground_true) * lambda_L1
    # combine loss and calculate gradients
    loss_G = loss_G_GAN + loss_G_L1
    loss_sum[0] += loss_G_L1.item()
    loss_sum[1] += loss_G.item()

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()



    # a = netD(ground_true)
    # print(a.size())
    # loss = criterion_L1(pred, ground_true)+criterion_L2(pred, ground_true)
    # # loss = criterion_L2(pred, ground_true)
    # loss_sum += loss.item()

    # optimizer_G.zero_grad()
    # loss.backward()
    # optimizer_G.step()

    if (iter+1)%100 == 0:
        showresult(input_img[:,int((N+1)/2)*3:(int((N+1)/2)+1)*3,:,:], ground_true, pred,'./result_train.png')
    if (iter+1)%100 == 0:
        time_end=time.time()
        print('iter:',iter+1,' L1_loss:', round(loss_sum[0]/100,4),'G_loss:', round(loss_sum[1]/100,4),'time:',round((time_end-time_start)/100,4))
        if (iter+1)/100 >= 10:
            loss_plot[0].append(loss_sum[0]/100)
            loss_plot[1].append(loss_sum[1]/100)
            item_plot.append(iter+1)
            plt.plot(item_plot,loss_plot[0])
            plt.plot(item_plot,loss_plot[1])
            plt.savefig('./loss.png')
            plt.close()
        loss_sum = [0.,0.]

        #show test result
        # netG.eval()
        # input_img = np.zeros((SIZE,SIZE,3*N), dtype='uint8')
        # imgs = os.listdir('./test')
        # for i in range(0,N):
        #     # print('./dataset/'+video+'/mosaic/output_'+'%05d'%(img_index+i-int(N/2))+'.png')
        #     img = cv2.imread('./test/'+imgs[i])
        #     img = impro.resize(img,SIZE)
        #     input_img[:,:,i*3:(i+1)*3] = img
        # input_img = im2tensor(input_img,use_gpu)
        # ground_true = cv2.imread('./test/output_'+'%05d'%13+'.png')
        # ground_true = impro.resize(ground_true,SIZE)
        # ground_true = im2tensor(ground_true,use_gpu)
        # pred = netG(input_img)
        # showresult(input_img[:,int((N+1)/2)*3:(int((N+1)/2)+1)*3,:,:],pred,pred,'./result_test.png')
        
        netG.train()
        time_start=time.time()

    if (iter+1)%SAVE_FRE == 0:
        torch.save(netG.cpu().state_dict(),dir_checkpoint+'last_G.pth')
        torch.save(netD.cpu().state_dict(),dir_checkpoint+'last_D.pth')
        if use_gpu:
            netG.cuda()
            netD.cuda()
        f = open('./iter','w+')
        f.write(str(iter+1))
        f.close()
        # torch.save(netG.cpu().state_dict(),dir_checkpoint+'iter'+str(iter+1)+'.pth')
        print('network saved.')
