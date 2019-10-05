import cv2
import numpy as np
import os
import random
from .image_processing import resize,ch_one2three,mask_area

def addmosaic(img,mask,opt):
    if opt.mosaic_mod == 'random':
        img = addmosaic_random(img,mask)
    elif opt.mosaic_size == 0:
        img = addmosaic_autosize(img, mask, opt.mosaic_mod)
    else:
        img = addmosaic_normal(img,mask,opt.mosaic_size,opt.output_size,model = opt.mosaic_mod)
    return img

def addmosaic_normal(img,mask,n,out_size = 0,model = 'squa_avg',rect_rat = 1.6):
    n = int(n)
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

    elif model=='squa_mid':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                if mask[int(i*n+n/2),int(j*n+n/2)] == 255:
                    img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[i*n+int(n/2),j*n+int(n/2),:]

    elif model == 'squa_random':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                if mask[int(i*n+n/2),int(j*n+n/2)] == 255:
                    img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[int(i*n-n/2+n*random.random()),int(j*n-n/2+n*random.random()),:]

    elif model == 'squa_avg_circle_edge':
        for i in range(int(h/n)):
            for j in range(int(w/n)):
                img_mosaic[i*n:(i+1)*n,j*n:(j+1)*n,:]=img[i*n:(i+1)*n,j*n:(j+1)*n,:].mean(0).mean(0)
        mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
        mask = ch_one2three(mask)
        mask_inv = cv2.bitwise_not(mask)
        imgroi1 = cv2.bitwise_and(mask,img_mosaic)
        imgroi2 = cv2.bitwise_and(mask_inv,img)
        img_mosaic = cv2.add(imgroi1,imgroi2)

    elif model =='rect_avg':
        n_h=n
        n_w=int(n*rect_rat)
        for i in range(int(h/n_h)):
            for j in range(int(w/n_w)):
                if mask[int(i*n_h+n_h/2),int(j*n_w+n_w/2)] == 255:
                    img_mosaic[i*n_h:(i+1)*n_h,j*n_w:(j+1)*n_w,:]=img[i*n_h:(i+1)*n_h,j*n_w:(j+1)*n_w,:].mean(0).mean(0)
    
    return img_mosaic

def get_autosize(img,mask,area_type = 'normal'):
    h,w = img.shape[:2]
    mask = cv2.resize(mask,(w,h))
    alpha = np.min((w,h))/512
    try:
        if area_type == 'normal':
            area = mask_area(mask)
        elif area_type == 'bounding':
            w,h = cv2.boundingRect(mask)[2:]
            area = w*h
    except:
        area = 0
    area = area/(alpha*alpha)
    if area>50000:
        size = alpha*((area-50000)/50000+12)
    elif 20000<area<=50000:
        size = alpha*((area-20000)/30000+8)
    elif 5000<area<=20000:
        size = alpha*((area-5000)/20000+7)
    elif 0<=area<=5000:
        size = alpha*((area-0)/5000+6)
    else:
        pass
    return size

def addmosaic_autosize(img,mask,model,area_type = 'normal'):
    h,w = img.shape[:2]
    mask = cv2.resize(mask,(w,h))
    alpha = np.min((w,h))/512
    try:
        if area_type == 'normal':
            area = mask_area(mask)
        elif area_type == 'bounding':
            w,h = cv2.boundingRect(mask)[2:]
            area = w*h
    except:
        area = 0
    area = area/(alpha*alpha)
    if area>50000:
        img_mosaic = addmosaic_normal(img,mask,alpha*((area-50000)/50000+12),model = model)
    elif 20000<area<=50000:
        img_mosaic = addmosaic_normal(img,mask,alpha*((area-20000)/30000+8),model = model)
    elif 5000<area<=20000:
        img_mosaic = addmosaic_normal(img,mask,alpha*((area-5000)/20000+7),model = model)
    elif 0<=area<=5000:
        img_mosaic = addmosaic_normal(img,mask,alpha*((area-0)/5000+6),model = model)
    else:
        pass
    return img_mosaic

def addmosaic_random(img,mask,area_type = 'normal'):
    # img = resize(img,512)
    h,w = img.shape[:2]
    mask = cv2.resize(mask,(w,h))
    alpha = np.min((w,h))/512
    #area_avg=5925*4
    try:
        if area_type == 'normal':
            area = mask_area(mask)
        elif area_type == 'bounding':
            w,h = cv2.boundingRect(mask)[2:]
            area = w*h
    except:
        area = 0
    area = area/(alpha*alpha)
    if area>50000:
        img_mosaic = random_mod(img,mask,alpha*random.uniform(16,30))
    elif 20000<area<=50000:
        img_mosaic = random_mod(img,mask,alpha*random.uniform(12,20))
    elif 5000<area<=20000:
        img_mosaic = random_mod(img,mask,alpha*random.uniform(8,15))
    elif 0<=area<=5000:
        img_mosaic = random_mod(img,mask,alpha*random.uniform(4,10))
    else:
        pass
    return img_mosaic

def random_mod(img,mask,n):
    ran=random.random()
    if ran < 0.2:
        img = addmosaic_normal(img,mask,n,model = 'squa_mid')
    if 0.2 <= ran < 0.4:
        img = addmosaic_normal(img,mask,n,model = 'squa_avg')
    elif 0.4 <= ran <0.6:
        img = addmosaic_normal(img,mask,n,model = 'squa_avg_circle_edge')
    else:
        img = addmosaic_normal(img,mask,n,model = 'rect_avg')
    return img