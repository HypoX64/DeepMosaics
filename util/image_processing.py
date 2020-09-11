import cv2
import numpy as np
import random

import platform

system_type = 'Linux'
if 'Windows' in platform.platform():
    system_type = 'Windows'

DCT_Q = np.array([[8,16,19,22,26,27,29,34],
                [16,16,22,24,27,29,34,37],
                [19,22,26,27,29,34,34,38],
                [22,22,26,27,29,34,37,40],
                [22,26,27,29,32,35,40,48],
                [26,27,29,32,35,40,48,58],
                [26,27,29,34,38,46,56,59],
                [27,29,35,38,46,56,69,83]])

def imread(file_path,mod = 'normal',loadsize = 0):
    '''
    mod:  'normal' | 'gray' | 'all'
    loadsize: 0->original
    '''
    if system_type == 'Linux':
        if mod == 'normal':
            img = cv2.imread(file_path,1)
        elif mod == 'gray':
            img = cv2.imread(file_path,0)
        elif mod == 'all':
            img = cv2.imread(file_path,-1)
    
    #In windows, for chinese path, use cv2.imdecode insteaded.
    #It will loss EXIF, I can't fix it
    else: 
        if mod == 'normal':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),1)
        elif mod == 'gray':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),0)
        elif mod == 'all':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            
    if loadsize != 0:
        img = resize(img, loadsize, interpolation=cv2.INTER_CUBIC)

    return img

def imwrite(file_path,img):
    '''
    in other to save chinese path images in windows,
    this fun just for save final output images
    '''
    if system_type == 'Linux':
        cv2.imwrite(file_path, img)
    else:
        cv2.imencode('.jpg', img)[1].tofile(file_path)

def resize(img,size,interpolation=cv2.INTER_LINEAR):
    '''
    cv2.INTER_NEAREST      最邻近插值点法
    cv2.INTER_LINEAR        双线性插值法
    cv2.INTER_AREA         邻域像素再取样插补
    cv2.INTER_CUBIC        双立方插补，4*4大小的补点
    cv2.INTER_LANCZOS4     8x8像素邻域的Lanczos插值
    '''
    h, w = img.shape[:2]
    if np.min((w,h)) ==size:
        return img
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size),interpolation=interpolation)
    else:
        res = cv2.resize(img,(size, int(size*h/w)),interpolation=interpolation)
    return res

def resize_like(img,img_like):
    h, w = img_like.shape[:2]
    img = cv2.resize(img, (w,h))
    return img

def ch_one2three(img):
    res = cv2.merge([img, img, img])
    return res

def color_adjust(img,alpha=0,beta=0,b=0,g=0,r=0,ran = False):
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
        alpha = random.uniform(-0.1,0.1)
        beta  = random.uniform(-0.1,0.1)
        b     = random.uniform(-0.05,0.05)
        g     = random.uniform(-0.05,0.05)
        r     = random.uniform(-0.05,0.05)
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

def block_dct_and_idct(g,QQF,QQF_16):
    return cv2.idct(np.round(16.0*cv2.dct(g)/QQF)*QQF_16)

def image_dct_and_idct(I,QF):
    h,w = I.shape
    QQF = DCT_Q*QF
    QQF_16 = QQF/16.0
    for i in range(h//8):
        for j in range(w//8):
            I[i*8:(i+1)*8,j*8:(j+1)*8] = cv2.idct(np.round(16.0*cv2.dct(I[i*8:(i+1)*8,j*8:(j+1)*8])/QQF)*QQF_16)
            #I[i*8:(i+1)*8,j*8:(j+1)*8] = block_dct_and_idct(I[i*8:(i+1)*8,j*8:(j+1)*8],QQF,QQF_16)
    return I

def dctblur(img,Q):
    '''
    Q: 1~20, 1->best
    '''
    h,w = img.shape[:2]
    img = img[:8*(h//8),:8*(w//8)]
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = image_dct_and_idct(img, Q)
    if img.ndim == 3:
        h,w,ch = img.shape
        for i in range(ch):
            img[:,:,i] = image_dct_and_idct(img[:,:,i], Q)
    return (np.clip(img,0,255)).astype(np.uint8)
    
def find_mostlikely_ROI(mask):
    contours,hierarchy=cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        areas = []
        for contour in contours:
            areas.append(cv2.contourArea(contour))
        index = areas.index(max(areas))
        mask = np.zeros_like(mask)
        mask = cv2.fillPoly(mask,[contours[index]],(255))
    return mask

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

def mask_threshold(mask,ex_mun,threshold):
    mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)[1]
    mask = cv2.blur(mask, (ex_mun, ex_mun))
    mask = cv2.threshold(mask,threshold/5,255,cv2.THRESH_BINARY)[1]
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


def Q_lapulase(resImg):
    '''
    Evaluate image quality
    score > 20   normal
    score > 50   clear
    '''
    img2gray = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
    img2gray = resize(img2gray,512)
    res = cv2.Laplacian(img2gray, cv2.CV_64F)
    score = res.var()
    return score

def replace_mosaic(img_origin,img_fake,mask,x,y,size,no_feather):
    img_fake = cv2.resize(img_fake,(size*2,size*2),interpolation=cv2.INTER_LANCZOS4)
    if no_feather:
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

        mask = cv2.resize(mask,(img_origin.shape[1],img_origin.shape[0]))
        mask = ch_one2three(mask)
        
        mask = (cv2.blur(mask, (eclosion_num, eclosion_num)))
        mask_tmp = np.zeros_like(mask)
        mask_tmp[y-size:y+size,x-size:x+size] = mask[y-size:y+size,x-size:x+size]# Fix edge overflow
        mask = mask_tmp/255.0

        img_tmp = np.zeros(img_origin.shape)
        img_tmp[y-size:y+size,x-size:x+size]=img_fake
        img_result = img_origin.copy()
        img_result = (img_origin*(1-mask)+img_tmp*mask).astype('uint8')

    return img_result