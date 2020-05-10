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
        img = addmosaic_base(img,mask,opt.mosaic_size,opt.output_size,model = opt.mosaic_mod)
    return img

def addmosaic_base(img,mask,n,out_size = 0,model = 'squa_avg',rect_rat = 1.6,feather=0,start_point=[0,0]):
    '''
    img: input image
    mask: input mask
    n: mosaic size
    out_size: output size  0->original
    model : squa_avg squa_mid squa_random squa_avg_circle_edge rect_avg
    rect_rat: if model==rect_avg , mosaic w/h=rect_rat
    feather : feather size, -1->no 0->auto
    start_point : [0,0], please not input this parameter
    '''
    n = int(n)
    
    h_start = np.clip(start_point[0], 0, n)
    w_start = np.clip(start_point[1], 0, n)
    pix_mid_h = n//2+h_start
    pix_mid_w = n//2+w_start
    h, w = img.shape[:2]
    h_step = (h-h_start)//n
    w_step = (w-w_start)//n
    if out_size:
        img = resize(img,out_size)      
    if mask.shape[0] != h:
        mask = cv2.resize(mask,(w,h))
    img_mosaic = img.copy()

    if model=='squa_avg':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                           img[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:].mean(axis=(0,1))

    elif model=='squa_mid':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                           img[i*n+n//2+h_start,j*n+n//2+w_start,:]

    elif model == 'squa_random':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                    img[h_start+int(i*n-n/2+n*random.random()),w_start+int(j*n-n/2+n*random.random()),:]

    elif model == 'squa_avg_circle_edge':
        for i in range(h_step):
            for j in range(w_step):
                img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                       img[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:].mean(axis=(0,1))
        mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
        _mask = ch_one2three(mask)
        mask_inv = cv2.bitwise_not(_mask)
        imgroi1 = cv2.bitwise_and(_mask,img_mosaic)
        imgroi2 = cv2.bitwise_and(mask_inv,img)
        img_mosaic = cv2.add(imgroi1,imgroi2)

    elif model =='rect_avg':
        n_h = n
        n_w = int(n*rect_rat)
        n_h_half = n_h//2+h_start
        n_w_half = n_w//2+w_start
        for i in range((h-h_start)//n_h):
            for j in range((w-w_start)//n_w):
                if mask[i*n_h+n_h_half,j*n_w+n_w_half]:
                    img_mosaic[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:]=\
                           img[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:].mean(axis=(0,1))
    
    if feather != -1:
        if feather==0:
            mask = (cv2.blur(mask, (n, n)))
        else:
            mask = (cv2.blur(mask, (feather, feather)))
        mask = mask/255.0
        for i in range(3):img_mosaic[:,:,i] = (img[:,:,i]*(1-mask)+img_mosaic[:,:,i]*mask)
        img_mosaic = img_mosaic.astype(np.uint8)
    
    return img_mosaic

def get_autosize(img,mask,area_type = 'normal'):
    h,w = img.shape[:2]
    size = np.min([h,w])
    mask = resize(mask,size)
    alpha = size/512
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

def get_random_parameter(img,mask):
    # mosaic size
    p = np.array([0.5,0.5])
    mod = np.random.choice(['normal','bounding'], p = p.ravel())
    mosaic_size = get_autosize(img,mask,area_type = mod)
    mosaic_size = int(mosaic_size*random.uniform(0.9,2.5))

    # mosaic mod
    p = np.array([0.25, 0.3, 0.45])
    mod = np.random.choice(['squa_mid','squa_avg','rect_avg'], p = p.ravel())

    # rect_rat for rect_avg
    rect_rat = random.uniform(1.1,1.6)
    
    # feather size
    feather = -1
    if random.random()<0.7:
        feather = int(mosaic_size*random.uniform(0,1.5))

    return mosaic_size,mod,rect_rat,feather


def addmosaic_autosize(img,mask,model,area_type = 'normal'):
    mosaic_size = get_autosize(img,mask,area_type = 'normal')
    img_mosaic = addmosaic_base(img,mask,mosaic_size,model = model)
    return img_mosaic

def addmosaic_random(img,mask):
    mosaic_size,mod,rect_rat,feather = get_random_parameter(img,mask)
    img_mosaic = addmosaic_base(img,mask,mosaic_size,model = mod,rect_rat=rect_rat,feather=feather)
    return img_mosaic

def get_random_startpos(num,bisa_p,bisa_max,bisa_max_part):
    pos = np.zeros((num,2), dtype=np.int64)
    if random.random()<bisa_p:
        indexs = random.sample((np.linspace(1,num-1,num-1,dtype=np.int64)).tolist(), random.randint(1, bisa_max_part))
        indexs.append(0)
        indexs.append(num)
        indexs.sort()
        for i in range(len(indexs)-1):
            pos[indexs[i]:indexs[i+1]] = [random.randint(0,bisa_max),random.randint(0,bisa_max)]
    return pos