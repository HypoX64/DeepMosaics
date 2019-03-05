import cv2
import numpy as np
import os
import random
from .image_processing import resize,channel_one2three


def addmosaic(img,mask,n,out_size = 0,model = 'squa_avg'):
    if out_size:
        img = resize(img,out_size)      
    h, w = img.shape[:2]
    mask = cv2.resize(mask,(w,h))

    img_mosaic=img.copy()

    if model=='squa_avg':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                if mask[int(i*n+n/2),int(j*n+n/2)] == 255:
                    img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[i*n:(i+1)*n,j*n:(j+1)*n,:].mean(0).mean(0)

    elif model == 'squa_random':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                if mask[int(i*n+n/2),int(j*n+n/2)] == 255:
                    img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[int(i*n-n/2+n*random.random()),int(j*n-n/2+n*random.random()),:]

    elif model == 'squa_avg_circle_edge':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[i*n:(i+1)*n,j*n:(j+1)*n,:].mean(0).mean(0)
        mask = channel_one2three(mask)
        mask_inv = cv2.bitwise_not(mask)
        imgroi1 = cv2.bitwise_and(mask,img_mosaic)
        imgroi2 = cv2.bitwise_and(mask_inv,img)
        img_mosaic = cv2.add(imgroi1,imgroi2)

    elif model =='rect_avg':
        rect_ratio=1+0.6*random.random()
        n_h=n
        n_w=int(n*rect_ratio)
        for i in range(int(h/n_h)):
            for j in range(int(w/n_w)):
                if mask[int(i*n_h+n_h/2),int(j*n_w+n_w/2)] == 255:
                    img_mosaic[i*n_h:(i+1)*n_h,j*n_w:(j+1)*n_w,:]=img[i*n_h:(i+1)*n_h,j*n_w:(j+1)*n_w,:].mean(0).mean(0)
    
    return img_mosaic

def random_mosaic_mod(img,mask,n):
    ran=random.random()
    if ran < 0.1:
        img = addmosaic(img,mask,n,model = 'squa_random')
    if 0.1 <= ran < 0.3:
        img = addmosaic(img,mask,n,model = 'squa_avg')
    elif 0.3 <= ran <0.5:
        img = addmosaic(img,mask,n,model = 'squa_avg_circle_edge')
    else:
        img = addmosaic(img,mask,n,model = 'rect_avg')
    return img

def random_mosaic(img,mask):
    img = resize(img,512)
    h,w = img.shape[:2]
    mask = cv2.resize(mask,(w,h))
    #area_avg=5925*4
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        area = cv2.contourArea(contours[0])
    except:
        area = 0
    if area>50000:
        img_mosaic = random_mosaic_mod(img,mask,random.randint(14,26))
    elif 20000<area<=50000:
        img_mosaic = random_mosaic_mod(img,mask,random.randint(10,18))
    elif 5000<area<=20000:
        img_mosaic = random_mosaic_mod(img,mask,random.randint(8,14))
    elif 0<=area<=5000:
        img_mosaic = random_mosaic_mod(img,mask,random.randint(4,8))
    else:
        pass
    return img_mosaic
