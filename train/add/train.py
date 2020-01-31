import sys
import os
import random
import datetime

import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

import sys
sys.path.append("..")
sys.path.append("../..")
from util import mosaic,util,ffmpeg,filt,data
from util import image_processing as impro
from models import unet_model
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn

LR = 0.0002
EPOCHS = 100
BATCHSIZE = 16
LOADSIZE = 256
FINESIZE = 224
CONTINUE = True
use_gpu = True
SAVE_FRE = 1
MAX_LOAD = 30000



dir_img = './datasets/face/origin_image/'
dir_mask = './datasets/face/mask/'
dir_checkpoint = 'checkpoints/face/'


def Totensor(img,use_gpu=True):
    size=img.shape[0]
    img = torch.from_numpy(img).float()
    if use_gpu:
        img = img.cuda()
    return img


def Toinputshape(imgs,masks,finesize,test_flag = False):
    batchsize = len(imgs)
    result_imgs=[];result_masks=[]
    for i in range(batchsize):
        # print(imgs[i].shape,masks[i].shape)
        img,mask = data.random_transform_image(imgs[i], masks[i], finesize, test_flag)
        # print(img.shape,mask.shape)
        mask = (mask.reshape(1,finesize,finesize)/255.0)
        img = (img.transpose((2, 0, 1))/255.0)
        result_imgs.append(img)
        result_masks.append(mask)
    result_imgs = np.array(result_imgs)
    result_masks  = np.array(result_masks)
    return result_imgs,result_masks

def batch_generator(images,masks,batchsize):
    dataset_images = []
    dataset_masks = []

    for i in range(int(len(images)/batchsize)):
        dataset_images.append(images[i*batchsize:(i+1)*batchsize])
        dataset_masks.append(masks[i*batchsize:(i+1)*batchsize])
    if len(images)%batchsize != 0:
        dataset_images.append(images[len(images)-len(images)%batchsize:])
        dataset_masks.append(masks[len(images)-len(images)%batchsize:])

    return dataset_images,dataset_masks

def loadimage(dir_img,dir_mask,loadsize,eval_p):
    t1 = datetime.datetime.now()
    imgnames = os.listdir(dir_img)
    # imgnames = imgnames[:100]   
    random.shuffle(imgnames)
    imgnames = imgnames[:MAX_LOAD]
    print('load images:',len(imgnames))
    imgnames = (f[:-4] for f in imgnames)
    images = []
    masks = []
    for imgname in imgnames:
        img = impro.imread(dir_img+imgname+'.jpg')
        mask = impro.imread(dir_mask+imgname+'.png',mod = 'gray')
        img = impro.resize(img,loadsize)
        mask = impro.resize(mask,loadsize)
        images.append(img)
        masks.append(mask)
    train_images,train_masks = images[0:int(len(masks)*(1-eval_p))],masks[0:int(len(masks)*(1-eval_p))]
    eval_images,eval_masks = images[int(len(masks)*(1-eval_p)):len(masks)],masks[int(len(masks)*(1-eval_p)):len(masks)]
    t2 = datetime.datetime.now()
    print('load data cost time:',(t2 - t1).seconds,'s')
    return train_images,train_masks,eval_images,eval_masks


util.makedirs(dir_checkpoint)
print('loading data......')
train_images,train_masks,eval_images,eval_masks = loadimage(dir_img,dir_mask,LOADSIZE,0.2)
dataset_eval_images,dataset_eval_masks = batch_generator(eval_images,eval_masks,BATCHSIZE)
dataset_train_images,dataset_train_masks = batch_generator(train_images,train_masks,BATCHSIZE)


net = unet_model.UNet(n_channels = 3, n_classes = 1)


if CONTINUE:
    if not os.path.isfile(os.path.join(dir_checkpoint,'last.pth')):
        CONTINUE = False
        print('can not load last.pth, training on init weight.')
if CONTINUE:
    net.load_state_dict(torch.load(dir_checkpoint+'last.pth'))
if use_gpu:
    net.cuda()
    cudnn.benchmark = True


optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))

criterion = nn.BCELoss()
# criterion = nn.L1Loss()

print('begin training......')
for epoch in range(EPOCHS):
    random_save = random.randint(0, len(dataset_train_images))

    starttime = datetime.datetime.now()
    print('Epoch {}/{}.'.format(epoch + 1, EPOCHS))
    net.train()
    if use_gpu:
        net.cuda()
    epoch_loss = 0
    for i,(img,mask) in enumerate(zip(dataset_train_images,dataset_train_masks)):
        # print(epoch,i,img.shape,mask.shape)
        img,mask = Toinputshape(img, mask, FINESIZE)
        img = Totensor(img,use_gpu)
        mask = Totensor(mask,use_gpu)

        mask_pred = net(img)
        loss = criterion(mask_pred, mask)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            data.showresult(img,mask,mask_pred,os.path.join(dir_checkpoint,'result.png'),True)
        if  i == random_save:
            data.showresult(img,mask,mask_pred,os.path.join(dir_checkpoint,'epoch_'+str(epoch+1)+'.png'),True)

    # torch.cuda.empty_cache()
    # # net.eval()
    epoch_loss_eval = 0
    with torch.no_grad():
    #net.eval()
        for i,(img,mask) in enumerate(zip(dataset_eval_images,dataset_eval_masks)):
            # print(epoch,i,img.shape,mask.shape)
            img,mask = Toinputshape(img, mask, FINESIZE,test_flag=True)
            img = Totensor(img,use_gpu)
            mask = Totensor(mask,use_gpu)
            mask_pred = net(img)
            loss = criterion(mask_pred, mask)
            epoch_loss_eval += loss.item()
    # torch.cuda.empty_cache()

    endtime = datetime.datetime.now()
    print('--- Epoch train_loss: {0:.6f} eval_loss: {1:.6f} Cost time: {2:} s'.format(
        epoch_loss/len(dataset_train_images),
        epoch_loss_eval/len(dataset_eval_images),
        (endtime - starttime).seconds)),
    torch.save(net.cpu().state_dict(),dir_checkpoint+'last.pth')

    if (epoch+1)%SAVE_FRE == 0:
        torch.save(net.cpu().state_dict(),dir_checkpoint+'epoch'+str(epoch+1)+'.pth')
        
        print('network saved.')
