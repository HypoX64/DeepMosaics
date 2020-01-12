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

from unet import UNet

def resize(img,size):
    h, w = img.shape[:2]
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size))
    else:
        res = cv2.resize(img,(size, int(size*h/w)))
    return res


def Totensor(img,use_gpu=True):
    size=img.shape[0]
    img = torch.from_numpy(img).float()
    if use_gpu:
        img = img.cuda()
    return img

def random_color(img,random_num):
    for i in range(3): img[:,:,i]=np.clip(img[:,:,i].astype('int')+random.randint(-random_num,random_num),0,255).astype('uint8')
    bright = random.randint(-random_num*2,random_num*2)
    for i in range(3): img[:,:,i]=np.clip(img[:,:,i].astype('int')+bright,0,255).astype('uint8')
    return img

def Toinputshape(imgs,masks,finesize):
    batchsize = len(imgs)
    result_imgs=[];result_masks=[]
    for i in range(batchsize):
        # print(imgs[i].shape,masks[i].shape)
        img,mask = random_transform(imgs[i], masks[i], finesize)
        # print(img.shape,mask.shape)
        mask = mask[:,:,0].reshape(1,finesize,finesize)/255.0
        img = img.transpose((2, 0, 1))/255.0
        result_imgs.append(img)
        result_masks.append(mask)
    result_imgs = np.array(result_imgs)
    result_masks  = np.array(result_masks)
    return result_imgs,result_masks



def random_transform(img,mask,finesize):

    
    # randomsize = int(finesize*(1.2+0.2*random.random())+2)

    h,w = img.shape[:2]
    loadsize = min((h,w))
    a = (float(h)/float(w))*random.uniform(0.9, 1.1)

    if h<w:
        mask = cv2.resize(mask, (int(loadsize/a),loadsize))
        img = cv2.resize(img, (int(loadsize/a),loadsize))
    else:
        mask = cv2.resize(mask, (loadsize,int(loadsize*a)))
        img = cv2.resize(img, (loadsize,int(loadsize*a)))

    # mask = randomsize(mask,loadsize)
    # img = randomsize(img,loadsize)


    #random crop
    h,w = img.shape[:2]

    h_move = int((h-finesize)*random.random())
    w_move = int((w-finesize)*random.random())
    # print(h,w,h_move,w_move)
    img_crop = img[h_move:h_move+finesize,w_move:w_move+finesize]
    mask_crop = mask[h_move:h_move+finesize,w_move:w_move+finesize]
    
    #random rotation
    if random.random()<0.2:
        h,w = img_crop.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),90*int(4*random.random()),1)
        img = cv2.warpAffine(img_crop,M,(w,h))
        mask = cv2.warpAffine(mask_crop,M,(w,h))
    else:
        img,mask = img_crop,mask_crop

    #random color
    img=random_color(img, 15)
    
    #random flip
    if random.random()<0.5:
        if random.random()<0.5:
            img = cv2.flip(img,0)
            mask = cv2.flip(mask,0)
        else:
            img = cv2.flip(img,1)
            mask = cv2.flip(mask,1)
    return img,mask

def randomresize(img):
    size = np.min(img.shape[:2])
    img = resize(img, int(size*random.uniform(1,1.2)))
    img = resize(img, size)
    return img

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
    print('images num:',len(imgnames))
    random.shuffle(imgnames)
    imgnames = (f[:-4] for f in imgnames)
    images = []
    masks = []
    for imgname in imgnames:
        img = cv2.imread(dir_img+imgname+'.jpg')
        mask = cv2.imread(dir_mask+imgname+'.png')
        img = resize(img,loadsize)
        mask = resize(mask,loadsize)
        images.append(img)
        masks.append(mask)
    train_images,train_masks = images[0:int(len(masks)*(1-eval_p))],masks[0:int(len(masks)*(1-eval_p))]
    eval_images,eval_masks = images[int(len(masks)*(1-eval_p)):len(masks)],masks[int(len(masks)*(1-eval_p)):len(masks)]
    t2 = datetime.datetime.now()
    print('load data cost time:',(t2 - t1).seconds,'s')
    return train_images,train_masks,eval_images,eval_masks

def showresult(img,mask,mask_pred):
    img = (img.cpu().detach().numpy()*255)
    mask = (mask.cpu().detach().numpy()*255)
    mask_pred = (mask_pred.cpu().detach().numpy()*255)
    batchsize = img.shape[0]
    size = img.shape[3]
    ran =int(batchsize*random.random())
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] =img[ran].transpose((1, 2, 0))
    showimg[0:size,size:size*2,1] = mask[ran].reshape(size,size)
    showimg[0:size,size*2:size*3,1] = mask_pred[ran].reshape(size,size)

    # cv2.imshow("", showimg.astype('uint8'))
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     exit()
    cv2.imwrite('./result.jpg', showimg)



LR = 0.001
EPOCHS = 100
BATCHSIZE = 12
LOADSIZE = 144
FINESIZE = 128
CONTINUE = True
use_gpu = True
SAVE_FRE = 5
cudnn.benchmark = False

dir_img = './origin_image/'
dir_mask = './mask/'
dir_checkpoint = 'checkpoints/'

print('loading data......')
train_images,train_masks,eval_images,eval_masks = loadimage(dir_img,dir_mask,LOADSIZE,0.2)
dataset_eval_images,dataset_eval_masks = batch_generator(eval_images,eval_masks,BATCHSIZE)
dataset_train_images,dataset_train_masks = batch_generator(train_images,train_masks,BATCHSIZE)


net = UNet(n_channels = 3, n_classes = 1)


if CONTINUE:
    net.load_state_dict(torch.load(dir_checkpoint+'last.pth'))
if use_gpu:
    net.cuda()



# optimizer = optim.SGD(net.parameters(),
#                       lr=LR,
#                       momentum=0.9,
#                       weight_decay=0.0005)

optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

criterion = nn.BCELoss()
# criterion = nn.L1Loss()

print('begin training......')
for epoch in range(EPOCHS):

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

        if i%10 == 0:
            showresult(img,mask,mask_pred)

    # torch.cuda.empty_cache()
    # # net.eval()
    epoch_loss_eval = 0
    with torch.no_grad():
        for i,(img,mask) in enumerate(zip(dataset_eval_images,dataset_eval_masks)):
            # print(epoch,i,img.shape,mask.shape)
            img,mask = Toinputshape(img, mask, FINESIZE)
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
    # print('--- Epoch loss: {0:.6f}'.format(epoch_loss/i))
    # print('Cost time: ',(endtime - starttime).seconds,'s')
    if (epoch+1)%SAVE_FRE == 0:
        torch.save(net.cpu().state_dict(),dir_checkpoint+'epoch'+str(epoch+1)+'.pth')
        print('network saved.')
# torch.save(net.cpu().state_dict(),dir_checkpoint+'last.pth')        
# print('network saved.')

