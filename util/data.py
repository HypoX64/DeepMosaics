import random
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from . import image_processing as impro
from . import mosaic
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  
    ]  
)  

def tensor2im(image_tensor, imtype=np.uint8, gray=False, rgb2bgr = True ,is0_1 = False):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    
    if not is0_1:
        image_numpy = (image_numpy + 1)/2.0
    image_numpy = np.clip(image_numpy * 255.0,0,255) 

    # gray -> output 1ch
    if gray:
        h, w = image_numpy.shape[1:]
        image_numpy = image_numpy.reshape(h,w)
        return image_numpy.astype(imtype)

    # output 3ch
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose((1, 2, 0))  
    if rgb2bgr and not gray:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(imtype)


def im2tensor(image_numpy, imtype=np.uint8, gray=False,bgr2rgb = True, reshape = True, use_gpu = 0,  use_transform = True,is0_1 = True):
    
    if gray:
        h, w = image_numpy.shape
        image_numpy = (image_numpy/255.0-0.5)/0.5
        image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor = image_tensor.reshape(1,1,h,w)
    else:
        h, w ,ch = image_numpy.shape
        if bgr2rgb:
            image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
        if use_transform:
            image_tensor = transform(image_numpy)
        else:
            if is0_1:
                image_numpy = image_numpy/255.0
            else:
                image_numpy = (image_numpy/255.0-0.5)/0.5
            image_numpy = image_numpy.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor = image_tensor.reshape(1,ch,h,w)
    if use_gpu != -1:
        image_tensor = image_tensor.cuda()
    return image_tensor

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

def random_transform_video(src,target,finesize,N):
    #random blur
    if random.random()<0.2:
        h,w = src.shape[:2]
        src = src[:8*(h//8),:8*(w//8)]
        Q_ran = random.randint(1,15)
        src[:,:,:3*N] = impro.dctblur(src[:,:,:3*N],Q_ran)
        target = impro.dctblur(target,Q_ran)

    #random crop
    h,w = target.shape[:2]
    h_move = int((h-finesize)*random.random())
    w_move = int((w-finesize)*random.random())
    target = target[h_move:h_move+finesize,w_move:w_move+finesize,:]
    src = src[h_move:h_move+finesize,w_move:w_move+finesize,:]

    #random flip
    if random.random()<0.5:
        src = src[:,::-1,:]
        target = target[:,::-1,:]

    #random color
    alpha = random.uniform(-0.1,0.1)
    beta  = random.uniform(-0.1,0.1)
    b     = random.uniform(-0.05,0.05)
    g     = random.uniform(-0.05,0.05)
    r     = random.uniform(-0.05,0.05)
    for i in range(N):
        src[:,:,i*3:(i+1)*3] = impro.color_adjust(src[:,:,i*3:(i+1)*3],alpha,beta,b,g,r)
    target = impro.color_adjust(target,alpha,beta,b,g,r)

    #random resize blur
    if random.random()<0.5:
        interpolations = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
        size_ran = random.uniform(0.7,1.5)
        interpolation_up = interpolations[random.randint(0,2)]
        interpolation_down =interpolations[random.randint(0,2)]

        tmp = cv2.resize(src[:,:,:3*N], (int(finesize*size_ran),int(finesize*size_ran)),interpolation=interpolation_up)
        src[:,:,:3*N] = cv2.resize(tmp, (finesize,finesize),interpolation=interpolation_down)

        tmp = cv2.resize(target, (int(finesize*size_ran),int(finesize*size_ran)),interpolation=interpolation_up)
        target = cv2.resize(tmp, (finesize,finesize),interpolation=interpolation_down)

    return src,target

def random_transform_single(img,out_shape):
    out_h,out_w = out_shape
    img = cv2.resize(img,(int(out_w*random.uniform(1.1, 1.5)),int(out_h*random.uniform(1.1, 1.5))))
    h,w = img.shape[:2]
    h_move = int((h-out_h)*random.random())
    w_move = int((w-out_w)*random.random())
    img = img[h_move:h_move+out_h,w_move:w_move+out_w]
    if random.random()<0.5:
        if random.random()<0.5:
            img = img[:,::-1]
        else:
            img = img[::-1,:]
    if img.shape[0] != out_h or img.shape[1]!= out_w :
        img = cv2.resize(img,(out_w,out_h))
    return img

def random_transform_image(img,mask,finesize,test_flag = False):
    #random scale
    if random.random()<0.5:
        h,w = img.shape[:2]
        loadsize = min((h,w))
        a = (float(h)/float(w))*random.uniform(0.9, 1.1)
        if h<w:
            mask = cv2.resize(mask, (int(loadsize/a),loadsize))
            img = cv2.resize(img, (int(loadsize/a),loadsize))
        else:
            mask = cv2.resize(mask, (loadsize,int(loadsize*a)))
            img = cv2.resize(img, (loadsize,int(loadsize*a)))

    #random crop
    h,w = img.shape[:2]
    h_move = int((h-finesize)*random.random())
    w_move = int((w-finesize)*random.random())
    img_crop = img[h_move:h_move+finesize,w_move:w_move+finesize]
    mask_crop = mask[h_move:h_move+finesize,w_move:w_move+finesize]

    if test_flag:
        return img_crop,mask_crop
    
    #random rotation
    if random.random()<0.2:
        h,w = img_crop.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),90*int(4*random.random()),1)
        img = cv2.warpAffine(img_crop,M,(w,h))
        mask = cv2.warpAffine(mask_crop,M,(w,h))
    else:
        img,mask = img_crop,mask_crop

    #random color
    img = impro.color_adjust(img,ran=True)

    #random flip
    if random.random()<0.5:
        if random.random()<0.5:
            img = img[:,::-1,:]
            mask = mask[:,::-1]
        else:
            img = img[::-1,:,:]
            mask = mask[::-1,:]

    #random blur
    if random.random()<0.5:
        img = impro.dctblur(img,random.randint(1,15))
        
        # interpolations = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
        # size_ran = random.uniform(0.7,1.5)
        # img = cv2.resize(img, (int(finesize*size_ran),int(finesize*size_ran)),interpolation=interpolations[random.randint(0,2)])
        # img = cv2.resize(img, (finesize,finesize),interpolation=interpolations[random.randint(0,2)])
    
    #check shape
    if img.shape[0]!= finesize or img.shape[1]!= finesize or mask.shape[0]!= finesize or mask.shape[1]!= finesize:
        img = cv2.resize(img,(finesize,finesize))
        mask = cv2.resize(mask,(finesize,finesize))
        print('warning! shape error.')
    return img,mask


def load_train_video(videoname,img_index,opt):
    N = opt.N    
    input_img = np.zeros((opt.loadsize,opt.loadsize,3*N+1), dtype='uint8')
    # this frame
    this_mask = impro.imread(os.path.join(opt.dataset,videoname,'mask','%05d'%(img_index)+'.png'),'gray',loadsize=opt.loadsize)
    input_img[:,:,-1] = this_mask
    #print(os.path.join(opt.dataset,videoname,'origin_image','%05d'%(img_index)+'.jpg'))
    ground_true =  impro.imread(os.path.join(opt.dataset,videoname,'origin_image','%05d'%(img_index)+'.jpg'),loadsize=opt.loadsize)
    mosaic_size,mod,rect_rat,feather = mosaic.get_random_parameter(ground_true,this_mask)
    start_pos = mosaic.get_random_startpos(num=N,bisa_p=0.3,bisa_max=mosaic_size,bisa_max_part=3)
    # merge other frame
    for i in range(0,N):
        img =  impro.imread(os.path.join(opt.dataset,videoname,'origin_image','%05d'%(img_index+i-int(N/2))+'.jpg'),loadsize=opt.loadsize)
        mask = impro.imread(os.path.join(opt.dataset,videoname,'mask','%05d'%(img_index+i-int(N/2))+'.png'),'gray',loadsize=opt.loadsize)
        img_mosaic = mosaic.addmosaic_base(img, mask, mosaic_size,model = mod,rect_rat=rect_rat,feather=feather,start_point=start_pos[i])
        input_img[:,:,i*3:(i+1)*3] = img_mosaic
    # to tensor
    input_img,ground_true = random_transform_video(input_img,ground_true,opt.finesize,N)
    input_img = im2tensor(input_img,bgr2rgb=False,use_gpu=-1,use_transform = False,is0_1=False)
    ground_true = im2tensor(ground_true,bgr2rgb=False,use_gpu=-1,use_transform = False,is0_1=False)
    
    return input_img,ground_true


def showresult(img1,img2,img3,name,is0_1 = False):
    size = img1.shape[3]
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] = tensor2im(img1,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size:size*2] = tensor2im(img2,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size*2:size*3] = tensor2im(img3,rgb2bgr = False, is0_1 = is0_1)
    cv2.imwrite(name, showimg)
