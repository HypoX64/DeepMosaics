'''
https://github.com/sonack/GFRNet_pytorch_new
'''
import random
import cv2
import numpy as np

def gaussian_blur(img, sigma=3, size=13):
    if sigma > 0:
        if isinstance(size, int):
            size = (size, size)
        img = cv2.GaussianBlur(img, size, sigma)
    return img

def down(img, scale, shape):
    if scale > 1:
        h, w, _ = shape
        scaled_h, scaled_w = int(h / scale), int(w / scale)
        img = cv2.resize(img, (scaled_w, scaled_h), interpolation = cv2.INTER_CUBIC)
    return img

def up(img, scale, shape):
    if scale > 1:
        h, w, _ = shape
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
    return img
    
def awgn(img, level):
    if level > 0:
        noise = np.random.randn(*img.shape) * level
        img = (img + noise).clip(0,255).astype(np.uint8)
    return img

def jpeg_compressor(img,quality):
    if quality > 0:    # 0 indicating no lossy compression (i.e losslessly compression)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img = cv2.imdecode(cv2.imencode('.jpg', img, encode_param)[1], 1)
    return img

def get_random_degenerate_params(mod='strong'):
    '''
    mod : strong | only_downsample | only_4x | weaker_1 | weaker_2
    '''
    params = {}
    gaussianBlur_size_list = list(range(3,14,2))

    if mod == 'strong':
        gaussianBlur_sigma_list = [1 + x for x in range(3)]
        gaussianBlur_sigma_list += [0]
        downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
        awgn_level_list = list(range(1, 8, 1))        
        jpeg_quality_list = list(range(10, 41, 1))
        jpeg_quality_list += int(len(jpeg_quality_list) * 0.33) * [0]
    
    elif mod == 'only_downsample':
        gaussianBlur_sigma_list = [0]
        downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
        awgn_level_list = [0]
        jpeg_quality_list = [0]
    
    elif mod == 'only_4x':
        gaussianBlur_sigma_list = [0]
        downsample_scale_list = [4]
        awgn_level_list = [0]
        jpeg_quality_list = [0]

    elif mod == 'weaker_1':   # 0.5 trigger prob
        gaussianBlur_sigma_list = [1 + x for x in range(3)]
        gaussianBlur_sigma_list += int(len(gaussianBlur_sigma_list)) * [0] # 1/2 trigger this degradation
        
        downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
        downsample_scale_list += int(len(downsample_scale_list)) * [1]
        
        awgn_level_list = list(range(1, 8, 1))
        awgn_level_list += int(len(awgn_level_list)) * [0]
        
        jpeg_quality_list = list(range(10, 41, 1))
        jpeg_quality_list += int(len(jpeg_quality_list)) * [0]

    elif mod == 'weaker_2':    # weaker than weaker_1, jpeg [20,40]
        gaussianBlur_sigma_list = [1 + x for x in range(3)]
        gaussianBlur_sigma_list += int(len(gaussianBlur_sigma_list)) * [0] # 1/2 trigger this degradation
        
        downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
        downsample_scale_list += int(len(downsample_scale_list)) * [1]
        
        awgn_level_list = list(range(1, 8, 1))
        awgn_level_list += int(len(awgn_level_list)) * [0]
        
        jpeg_quality_list = list(range(20, 41, 1))
        jpeg_quality_list += int(len(jpeg_quality_list)) * [0]
    
    params['blur_sigma'] = random.choice(gaussianBlur_sigma_list)
    params['blur_size'] = random.choice(gaussianBlur_size_list)
    params['updown_scale'] = random.choice(downsample_scale_list)
    params['awgn_level'] = random.choice(awgn_level_list)
    params['jpeg_quality'] = random.choice(jpeg_quality_list)

    return params

def degradate(img,params,jpeg_last = True):
    shape = img.shape
    if not params:
        params = get_random_degenerate_params('original')
        
    if jpeg_last:
        img = gaussian_blur(img,params['blur_sigma'],params['blur_size'])
        img = down(img,params['updown_scale'],shape)
        img = awgn(img,params['awgn_level'])
        img = up(img,params['updown_scale'],shape)
        img = jpeg_compressor(img,params['jpeg_quality'])
    else:
        img = gaussian_blur(img,params['blur_sigma'],params['blur_size'])
        img = down(img,params['updown_scale'],shape)
        img = awgn(img,params['awgn_level'])
        img = jpeg_compressor(img,params['jpeg_quality'])
        img = up(img,params['updown_scale'],shape)

    return img