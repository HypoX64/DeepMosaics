import random
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from .image_processing import color_adjust

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  
    ]  
)  

def tensor2im(image_tensor, imtype=np.uint8, gray=False, rgb2bgr = True ,is0_1 = False):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    # if gray:
    #     image_numpy = (image_numpy+1.0)/2.0 * 255.0
    # else:
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = image_numpy.transpose((1, 2, 0))

    if not is0_1:
        image_numpy = (image_numpy + 1)/2.0
    image_numpy = np.clip(image_numpy * 255.0,0,255)  
    if rgb2bgr and not gray:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(imtype)


def im2tensor(image_numpy, imtype=np.uint8, gray=False,bgr2rgb = True, reshape = True, use_gpu = True,  use_transform = True,is0_1 = True):
    
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
    if use_gpu:
        image_tensor = image_tensor.cuda()
    return image_tensor


def random_transform_video(src,target,finesize,N):

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
    alpha = random.uniform(-0.3,0.3)
    beta  = random.uniform(-0.2,0.2)
    b     = random.uniform(-0.05,0.05)
    g     = random.uniform(-0.05,0.05)
    r     = random.uniform(-0.05,0.05)
    for i in range(N):
        src[:,:,i*3:(i+1)*3] = color_adjust(src[:,:,i*3:(i+1)*3],alpha,beta,b,g,r)
    target = color_adjust(target,alpha,beta,b,g,r)

    # random_num = 15
    # bright = random.randint(-random_num*2,random_num*2)
    # for i in range(N*3): src[:,:,i]=np.clip(src[:,:,i].astype('int')+bright,0,255).astype('uint8')
    # for i in range(3): target[:,:,i]=np.clip(target[:,:,i].astype('int')+bright,0,255).astype('uint8')

    return src,target


def random_transform_image(img,mask,finesize,test_flag = False):

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
    img = color_adjust(img,ran=True)
    # random_num = 15
    # for i in range(3): img[:,:,i]=np.clip(img[:,:,i].astype('int')+random.randint(-random_num,random_num),0,255).astype('uint8')
    # bright = random.randint(-random_num*2,random_num*2)
    # for i in range(3): img[:,:,i]=np.clip(img[:,:,i].astype('int')+bright,0,255).astype('uint8')

    #random flip
    if random.random()<0.5:
        if random.random()<0.5:
            img = img[:,::-1,:]
            mask = mask[:,::-1]
        else:
            img = img[::-1,:,:]
            mask = mask[::-1,:]

    #random blur
    if random.random()>0.5:
        size_ran = random.uniform(0.5,1.5)
        img = cv2.resize(img, (int(finesize*size_ran),int(finesize*size_ran)))
        img = cv2.resize(img, (finesize,finesize))
        #img = cv2.blur(img, (random.randint(1,3), random.randint(1,3)))
    return img,mask

def showresult(img1,img2,img3,name,is0_1 = False):
    size = img1.shape[3]
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] = tensor2im(img1,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size:size*2] = tensor2im(img2,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size*2:size*3] = tensor2im(img3,rgb2bgr = False, is0_1 = is0_1)
    cv2.imwrite(name, showimg)
