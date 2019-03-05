import numpy as np
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  
    ]  
)  

def tensor2im(image_tensor, imtype=np.uint8, rgb2bgr = True):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if rgb2bgr:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(imtype)


def im2tensor(image_numpy, imtype=np.uint8, bgr2rgb = True, reshape = True, use_gpu = True):
    if bgr2rgb:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    image_tensor = transform(image_numpy)
    if reshape:
        image_tensor=image_tensor.reshape(1,3,128,128)
    if use_gpu:
        image_tensor = image_tensor.cuda()
    return image_tensor