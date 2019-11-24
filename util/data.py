import numpy as np
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  
    ]  
)  

def tensor2im(image_tensor, imtype=np.uint8, gray=False, rgb2bgr = True):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    # if gray:
    #     image_numpy = (image_numpy+1.0)/2.0 * 255.0
    # else:
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if rgb2bgr and not gray:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(imtype)


def im2tensor(image_numpy, imtype=np.uint8, gray=False,bgr2rgb = True, reshape = True, use_gpu = True,  use_transform = True):
    
    if gray:
        h, w = image_numpy.shape
        image_numpy = (image_numpy/255.0-0.5)/0.5
        image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor=image_tensor.reshape(1,1,h,w)
    else:
        h, w ,ch = image_numpy.shape
        if bgr2rgb:
            image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
        if use_transform:
            image_tensor = transform(image_numpy)
        else:
            image_numpy = image_numpy/255.0
            image_numpy = image_numpy.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor=image_tensor.reshape(1,ch,h,w)
    if use_gpu:
        image_tensor = image_tensor.cuda()
    return image_tensor

# def im2tensor(image_numpy, use_gpu=False):
#     h, w ,ch = image_numpy.shape
#     image_numpy = image_numpy/255.0
#     image_numpy = image_numpy.transpose((2, 0, 1))
#     image_numpy = image_numpy.reshape(-1,ch,h,w)
#     img_tensor = torch.from_numpy(image_numpy).float()
#     if use_gpu:
#         img_tensor = img_tensor.cuda()
#     return img_tensor