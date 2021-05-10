import random
import os
from util.mosaic import get_random_parameter
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from . import image_processing as impro
from . import degradater

def to_tensor(data,gpu_id):
    data = torch.from_numpy(data)
    if gpu_id != '-1':
        data = data.cuda()
    return data

def normalize(data):
    '''
    normalize to -1 ~ 1
    '''
    return (data.astype(np.float32)/255.0-0.5)/0.5

def anti_normalize(data):
    return np.clip((data*0.5+0.5)*255,0,255).astype(np.uint8)

def tensor2im(image_tensor, gray=False, rgb2bgr = True ,is0_1 = False, batch_index=0):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[batch_index].cpu().float().numpy()
    
    if not is0_1:
        image_numpy = (image_numpy + 1)/2.0
    image_numpy = np.clip(image_numpy * 255.0,0,255) 

    # gray -> output 1ch
    if gray:
        h, w = image_numpy.shape[1:]
        image_numpy = image_numpy.reshape(h,w)
        return image_numpy.astype(np.uint8)

    # output 3ch
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose((1, 2, 0))  
    if rgb2bgr and not gray:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(np.uint8)


def im2tensor(image_numpy, gray=False,bgr2rgb = True, reshape = True, gpu_id = '-1',is0_1 = False):
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
        if is0_1:
            image_numpy = image_numpy/255.0
        else:
            image_numpy = (image_numpy/255.0-0.5)/0.5
        image_numpy = image_numpy.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor = image_tensor.reshape(1,ch,h,w)
    if gpu_id != '-1':
        image_tensor = image_tensor.cuda()
    return image_tensor

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

def random_transform_single_mask(img,out_shape):
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

def get_transform_params():
    crop_flag  = True
    rotat_flag = np.random.random()<0.2
    color_flag = True
    flip_flag  = np.random.random()<0.2
    degradate_flag  = np.random.random()<0.5
    flag_dict = {'crop':crop_flag,'rotat':rotat_flag,'color':color_flag,'flip':flip_flag,'degradate':degradate_flag}
    
    crop_rate = [np.random.random(),np.random.random()]
    rotat_rate = np.random.random()
    color_rate = [np.random.uniform(-0.05,0.05),np.random.uniform(-0.05,0.05),np.random.uniform(-0.05,0.05),
        np.random.uniform(-0.05,0.05),np.random.uniform(-0.05,0.05)]
    flip_rate = np.random.random()
    degradate_params = degradater.get_random_degenerate_params(mod='weaker_2')
    rate_dict = {'crop':crop_rate,'rotat':rotat_rate,'color':color_rate,'flip':flip_rate,'degradate':degradate_params}

    return {'flag':flag_dict,'rate':rate_dict}

def random_transform_single_image(img,finesize,params=None,test_flag = False):
    if params is None:
        params = get_transform_params()
    
    if params['flag']['degradate']:
        img = degradater.degradate(img,params['rate']['degradate'])

    if params['flag']['crop']:
        h,w = img.shape[:2]
        h_move = int((h-finesize)*params['rate']['crop'][0])
        w_move = int((w-finesize)*params['rate']['crop'][1])
        img = img[h_move:h_move+finesize,w_move:w_move+finesize]
    
    if test_flag:
        return img

    if params['flag']['rotat']:
        h,w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),90*int(4*params['rate']['rotat']),1)
        img = cv2.warpAffine(img,M,(w,h))

    if params['flag']['color']:
        img = impro.color_adjust(img,params['rate']['color'][0],params['rate']['color'][1],
            params['rate']['color'][2],params['rate']['color'][3],params['rate']['color'][4])

    if params['flag']['flip']:
        img = img[:,::-1]

    #check shape
    if img.shape[0]!= finesize or img.shape[1]!= finesize:
        img = cv2.resize(img,(finesize,finesize))
        print('warning! shape error.')
    return img

def random_transform_pair_image(img,mask,finesize,test_flag = False):
    params = get_transform_params()
    img = random_transform_single_image(img,finesize,params)
    params['flag']['degradate'] = False
    params['flag']['color'] = False
    mask = random_transform_single_image(mask,finesize,params)
    return img,mask

def showresult(img1,img2,img3,name,is0_1 = False):
    size = img1.shape[3]
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] = tensor2im(img1,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size:size*2] = tensor2im(img2,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size*2:size*3] = tensor2im(img3,rgb2bgr = False, is0_1 = is0_1)
    cv2.imwrite(name, showimg)
