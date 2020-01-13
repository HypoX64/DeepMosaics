import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import time

import sys
sys.path.append("..")
sys.path.append("../..")
from models import runmodel,loadmodel
from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro
from cores import Options
from models import pix2pix_model,video_model,unet_model
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn

N = 25
ITER = 10000000
LR = 0.0002
beta1 = 0.5
use_gpu = True
use_gan = False
use_L2 = False
CONTINUE =  False
lambda_L1 = 1.0#100.0
lambda_gan = 1.0

SAVE_FRE = 10000
start_iter = 0
finesize = 128
loadsize = int(finesize*1.1)

savename = 'MosaicNet_test'
dir_checkpoint = 'checkpoints/'+savename
util.makedirs(dir_checkpoint)

loss_sum = [0.,0.,0.,0.]
loss_plot = [[],[]]
item_plot = []

opt = Options().getparse()
videos = os.listdir('./dataset')
videos.sort()
lengths = []
for video in videos:
    video_images = os.listdir('./dataset/'+video+'/ori')
    lengths.append(len(video_images))
#unet_128
#resnet_9blocks
#netG = pix2pix_model.define_G(3*N+1, 3, 128, 'resnet_6blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
netG = video_model.HypoNet(3*N+1, 3)
# netG = unet_model.UNet(3*N+1, 3)
if use_gan:
    netD = pix2pix_model.define_D(3*2+1, 64, 'basic', n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[])
    #netD = pix2pix_model.define_D(3*2+1, 64, 'n_layers', n_layers_D=5, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[])

if CONTINUE:
    netG.load_state_dict(torch.load(os.path.join(dir_checkpoint,'last_G.pth')))
    if use_gan:
        netD.load_state_dict(torch.load(os.path.join(dir_checkpoint,'last_D.pth')))
    f = open(os.path.join(dir_checkpoint,'iter'),'r')
    start_iter = int(f.read())
    f.close()
if use_gpu:
    netG.cuda()
    if use_gan:
        netD.cuda()
    cudnn.benchmark = True

optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR,betas=(beta1, 0.999))
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()
if use_gan:
    optimizer_D = torch.optim.Adam(netG.parameters(), lr=LR,betas=(beta1, 0.999))
    criterionGAN = pix2pix_model.GANLoss(gan_mode='lsgan').cuda()

def random_transform(src,target,finesize):

    #random crop
    h,w = target.shape[:2]
    h_move = int((h-finesize)*random.random())
    w_move = int((w-finesize)*random.random())
    # print(h,w,h_move,w_move)
    target = target[h_move:h_move+finesize,w_move:w_move+finesize,:]
    src = src[h_move:h_move+finesize,w_move:w_move+finesize,:]

    #random flip
    if random.random()<0.5:
        src = src[:,::-1,:]
        target = target[:,::-1,:]

    #random color
    random_num = 15
    bright = random.randint(-random_num*2,random_num*2)
    for i in range(N*3): src[:,:,i]=np.clip(src[:,:,i].astype('int')+bright,0,255).astype('uint8')
    for i in range(3): target[:,:,i]=np.clip(target[:,:,i].astype('int')+bright,0,255).astype('uint8')

    return src,target


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
    cv2.imwrite(os.path.join(dir_checkpoint,name), showimg)


def loaddata():
    video_index = random.randint(0,len(videos)-1)
    video = videos[video_index]
    img_index = random.randint(N,lengths[video_index]- N)
    input_img = np.zeros((loadsize,loadsize,3*N+1), dtype='uint8')
    for i in range(0,N):
        # print('./dataset/'+video+'/mosaic/output_'+'%05d'%(img_index+i-int(N/2))+'.png')
        img = cv2.imread('./dataset/'+video+'/mosaic/output_'+'%05d'%(img_index+i-int(N/2))+'.png')
        img = impro.resize(img,loadsize)
        input_img[:,:,i*3:(i+1)*3] = img
    mask = cv2.imread('./dataset/'+video+'/mask/output_'+'%05d'%(img_index)+'.png',0)
    mask = impro.resize(mask,loadsize)
    mask = impro.mask_threshold(mask,15,128)
    input_img[:,:,-1] = mask

    ground_true = cv2.imread('./dataset/'+video+'/ori/output_'+'%05d'%(img_index)+'.png')
    ground_true = impro.resize(ground_true,loadsize)

    input_img,ground_true = random_transform(input_img,ground_true,finesize)
    input_img = data.im2tensor(input_img,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False)
    ground_true = data.im2tensor(ground_true,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False)
    
    return input_img,ground_true

print('preloading data, please wait 5s...')
input_imgs=[]
ground_trues=[]
load_cnt = 0
def preload():
    global load_cnt   
    while 1:
        try:
            input_img,ground_true = loaddata()
            input_imgs.append(input_img)
            ground_trues.append(ground_true)
            if len(input_imgs)>10:
                del(input_imgs[0])
                del(ground_trues[0])
            load_cnt += 1
            # time.sleep(0.1)
        except Exception as e:
            print("error:",e)

import threading
t = threading.Thread(target=preload,args=())  #t为新创建的线程
t.daemon = True
t.start()
while load_cnt < 10:
    time.sleep(0.1)

netG.train()
time_start=time.time()
print("Begin training...")
for iter in range(start_iter+1,ITER):

    # input_img,ground_true = loaddata()
    ran = random.randint(1, 8)
    input_img = input_imgs[ran]
    ground_true = ground_trues[ran]

    pred = netG(input_img)

    if use_gan:
        netD.train()
        # print(input_img[0,3*N,:,:].size())
        # print((input_img[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:]).size())
        real_A = torch.cat((input_img[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], input_img[:,-1,:,:].reshape(-1,1,finesize,finesize)), 1)
        fake_AB = torch.cat((real_A, pred), 1)
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = criterionGAN(pred_fake, False)

        real_AB = torch.cat((real_A, ground_true), 1)
        pred_real = netD(real_AB)
        loss_D_real = criterionGAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_sum[2] += loss_D_fake.item()
        loss_sum[3] += loss_D_real.item()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        netD.eval()

        # fake_AB = torch.cat((input_img[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], pred), 1)
        real_A = torch.cat((input_img[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], input_img[:,-1,:,:].reshape(-1,1,finesize,finesize)), 1)
        fake_AB = torch.cat((real_A, pred), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = criterionGAN(pred_fake, True)*lambda_gan
        # Second, G(A) = B
        if use_L2:
            loss_G_L1 = (criterion_L1(pred, ground_true)+criterion_L2(pred, ground_true)) * lambda_L1
        else:
            loss_G_L1 = criterion_L1(pred, ground_true) * lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_sum[0] += loss_G_L1.item()
        loss_sum[1] += loss_G_GAN.item()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    else:
        if use_L2:
            loss_G_L1 = (criterion_L1(pred, ground_true)+criterion_L2(pred, ground_true)) * lambda_L1
        else:
            loss_G_L1 = criterion_L1(pred, ground_true) * lambda_L1
        loss_sum[0] += loss_G_L1.item()

        optimizer_G.zero_grad()
        loss_G_L1.backward()
        optimizer_G.step()


    if (iter+1)%100 == 0:
        try:
            showresult(input_img[:,int((N-1)/2)*3:(int((N-1)/2)+1)*3,:,:], ground_true, pred,'result_train.png')
        except Exception as e:
            print(e)
     
    if (iter+1)%1000 == 0:
        time_end = time.time()
        if use_gan:
            print('iter:',iter+1,' L1_loss:', round(loss_sum[0]/1000,4),' G_loss:', round(loss_sum[1]/1000,4),
                ' D_f:',round(loss_sum[2]/1000,4),' D_r:',round(loss_sum[3]/1000,4),' time:',round((time_end-time_start)/1000,2))
            if (iter+1)/1000 >= 10:
                loss_plot[0].append(loss_sum[0]/1000)
                loss_plot[1].append(loss_sum[1]/1000)
                item_plot.append(iter+1)
                try:
                    plt.plot(item_plot,loss_plot[0])
                    plt.plot(item_plot,loss_plot[1])
                    plt.savefig(os.path.join(dir_checkpoint,'loss.png'))
                    plt.close()
                except Exception as e:
                    print("error:",e)
        else:
            print('iter:',iter+1,' L1_loss:',round(loss_sum[0]/1000,4),' time:',round((time_end-time_start)/1000,2))
            if (iter+1)/1000 >= 10:
                loss_plot[0].append(loss_sum[0]/1000)
                item_plot.append(iter+1)
                try:
                    plt.plot(item_plot,loss_plot[0])
                    plt.savefig(os.path.join(dir_checkpoint,'loss.png'))
                    plt.close()
                except Exception as e:
                    print("error:",e)
        loss_sum = [0.,0.,0.,0.]
        time_start=time.time()



    if (iter+1)%SAVE_FRE == 0:
        if iter+1 != SAVE_FRE:
            os.rename(os.path.join(dir_checkpoint,'last_G.pth'),os.path.join(dir_checkpoint,str(iter+1-SAVE_FRE)+'G.pth'))
        torch.save(netG.cpu().state_dict(),os.path.join(dir_checkpoint,'last_G.pth'))
        if use_gan:
            if iter+1 != SAVE_FRE:
                os.rename(os.path.join(dir_checkpoint,'last_D.pth'),os.path.join(dir_checkpoint,str(iter+1-SAVE_FRE)+'D.pth'))
            torch.save(netD.cpu().state_dict(),os.path.join(dir_checkpoint,'last_D.pth'))
        if use_gpu:
            netG.cuda()
            if use_gan:
                netD.cuda()
        f = open(os.path.join(dir_checkpoint,'iter'),'w+')
        f.write(str(iter+1))
        f.close()
        # torch.save(netG.cpu().state_dict(),dir_checkpoint+'iter'+str(iter+1)+'.pth')
        print('network saved.')

        #test
        netG.eval()
        result = np.zeros((finesize*2,finesize*4,3), dtype='uint8')
        test_names = os.listdir('./test')

        for cnt,test_name in enumerate(test_names,0):
            img_names = os.listdir(os.path.join('./test',test_name,'image'))
            input_img = np.zeros((finesize,finesize,3*N+1), dtype='uint8')
            img_names.sort()
            for i in range(0,N):
                img = impro.imread(os.path.join('./test',test_name,'image',img_names[i]))
                img = impro.resize(img,finesize)
                input_img[:,:,i*3:(i+1)*3] = img

            mask = impro.imread(os.path.join('./test',test_name,'mask.png'),'gray')
            mask = impro.resize(mask,finesize)
            mask = impro.mask_threshold(mask,15,128)
            input_img[:,:,-1] = mask
            result[0:finesize,finesize*cnt:finesize*(cnt+1),:] = input_img[:,:,int((N-1)/2)*3:(int((N-1)/2)+1)*3]
            input_img = data.im2tensor(input_img,bgr2rgb=False,use_gpu=opt.use_gpu,use_transform = False)
            pred = netG(input_img)
 
            pred = (pred.cpu().detach().numpy()*255)[0].transpose((1, 2, 0))
            result[finesize:finesize*2,finesize*cnt:finesize*(cnt+1),:] = pred

        cv2.imwrite(os.path.join(dir_checkpoint,str(iter+1)+'_test.png'), result)
        netG.train()