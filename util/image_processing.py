import cv2
import numpy as np
import random

def imread(file_path,mod = 'normal'):
    '''
    mod = 'normal' | 'gray' | 'all'
    '''
    if mod == 'normal':
        cv_img = cv2.imread(file_path)
    elif mod == 'gray':
        cv_img = cv2.imread(file_path,0)
    elif mod == 'all':
        cv_img = cv2.imread(file_path,-1)

    # # imread for chinese path in windows but no EXIF
    # cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def resize(img,size):
    h, w = img.shape[:2]
    if np.min((w,h)) ==size:
        return img
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size))
    else:
        res = cv2.resize(img,(size, int(size*h/w)))
    return res

def resize_like(img,img_like):
    h, w = img_like.shape[:2]
    img = cv2.resize(img, (w,h))
    return img

def ch_one2three(img):
    #zeros = np.zeros(img.shape[:2], dtype = "uint8")
    # ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    res = cv2.merge([img, img, img])
    return res

def color_adjust(img,alpha=1,beta=0,b=0,g=0,r=0,ran = False):
    '''
    g(x) = (1+α)g(x)+255*β, 
    g(x) = g(x[:+b*255,:+g*255,:+r*255])
    
    Args:
        img   : input image
        alpha : contrast
        beta  : brightness
        b     : blue hue
        g     : green hue
        r     : red hue
        ran   : if True, randomly generated color correction parameters
    Retuens:
        img   : output image
    '''
    img = img.astype('float')
    if ran:
        alpha = random.uniform(-0.2,0.2)
        beta  = random.uniform(-0.2,0.2)
        b     = random.uniform(-0.1,0.1)
        g     = random.uniform(-0.1,0.1)
        r     = random.uniform(-0.1,0.1)
    img = (1+alpha)*img+255.0*beta
    bgr = [b*255.0,g*255.0,r*255.0]
    for i in range(3): img[:,:,i]=img[:,:,i]+bgr[i]
    
    return (np.clip(img,0,255)).astype('uint8')

def makedataset(target_image,orgin_image):
    target_image = resize(target_image,256)
    orgin_image = resize(orgin_image,256)
    img = np.zeros((256,512,3), dtype = "uint8")
    w = orgin_image.shape[1]
    img[0:256,0:256] = target_image[0:256,int(w/2-256/2):int(w/2+256/2)]
    img[0:256,256:512] = orgin_image[0:256,int(w/2-256/2):int(w/2+256/2)]
    return img

def image2folat(img,ch):
    size=img.shape[0]
    if ch == 1:
        img = (img[:,:,0].reshape(1,size,size)/255.0).astype(np.float32)
    else:
        img = (img.transpose((2, 0, 1))/255.0).astype(np.float32)
    return img

def spiltimage(img,size = 128):
    h, w = img.shape[:2]
    # size = min(h,w)
    if w >= h:
        img1 = img[:,0:size]
        img2 = img[:,w-size:w]
    else:
        img1 = img[0:size,:]
        img2 = img[h-size:h,:]

    return img1,img2

def mergeimage(img1,img2,orgin_image,size = 128):
    h, w = orgin_image.shape[:2]
    new_img1 = np.zeros((h,w), dtype = "uint8")
    new_img2 = np.zeros((h,w), dtype = "uint8")

    # size = min(h,w)
    if w >= h:
        new_img1[:,0:size]=img1
        new_img2[:,w-size:w]=img2
    else:
        new_img1[0:size,:]=img1
        new_img2[h-size:h,:]=img2
    result_img = cv2.add(new_img1,new_img2)
    return result_img

def boundingSquare(mask,Ex_mul):
    # thresh = mask_threshold(mask,10,threshold)
    area = mask_area(mask)
    if area == 0 :
        return 0,0,0,0

    x,y,w,h = cv2.boundingRect(mask)
    
    center = np.array([int(x+w/2),int(y+h/2)])
    size = max(w,h)
    point0=np.array([x,y])
    point1=np.array([x+size,y+size])

    h, w = mask.shape[:2]
    if size*Ex_mul > min(h, w):
        size = min(h, w)
        halfsize = int(min(h, w)/2)
    else:
        size = Ex_mul*size
        halfsize = int(size/2)
        size = halfsize*2
    point0 = center - halfsize
    point1 = center + halfsize
    if point0[0]<0:
        point0[0]=0
        point1[0]=size
    if point0[1]<0:
        point0[1]=0
        point1[1]=size
    if point1[0]>w:
        point1[0]=w
        point0[0]=w-size
    if point1[1]>h:
        point1[1]=h
        point0[1]=h-size
    center = ((point0+point1)/2).astype('int')
    return center[0],center[1],halfsize,area

def mask_threshold(mask,blur,threshold):
    mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)[1]
    mask = cv2.blur(mask, (blur, blur))
    mask = cv2.threshold(mask,threshold/3,255,cv2.THRESH_BINARY)[1]
    return mask

def mask_area(mask):
    mask = cv2.threshold(mask,127,255,0)[1]
    # contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1] #for opencv 3.4
    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]#updata to opencv 4.0
    try:
        area = cv2.contourArea(contours[0])
    except:
        area = 0
    return area


def replace_mosaic(img_origin,img_fake,x,y,size,no_father):
    img_fake = resize(img_fake,size*2)
    if no_father:
        img_origin[y-size:y+size,x-size:x+size]=img_fake
        img_result = img_origin
    else:
        #color correction
        RGB_origin = img_origin[y-size:y+size,x-size:x+size].mean(0).mean(0)
        RGB_fake = img_fake.mean(0).mean(0)
        for i in range(3):img_fake[:,:,i] = np.clip(img_fake[:,:,i]+RGB_origin[i]-RGB_fake[i],0,255)      
        #eclosion
        eclosion_num = int(size/5)
        entad = int(eclosion_num/2+2)
        mask = np.zeros(img_origin.shape, dtype='uint8')
        mask = cv2.rectangle(mask,(x-size+entad,y-size+entad),(x+size-entad,y+size-entad),(255,255,255),-1)
        mask = (cv2.blur(mask, (eclosion_num, eclosion_num)))
        mask = mask/255.0

        img_tmp = np.zeros(img_origin.shape)
        img_tmp[y-size:y+size,x-size:x+size]=img_fake
        img_result = img_origin.copy()
        img_result = (img_origin*(1-mask)+img_tmp*mask).astype('uint8')
    return img_result