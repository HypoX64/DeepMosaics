import cv2
import numpy as np


def resize(img,size):
    h, w = img.shape[:2]
    if np.min((w,h)) ==size:
        return img
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size))
    else:
        res = cv2.resize(img,(size, int(size*h/w)))
    return res

def ch_one2three(img):
    #zeros = np.zeros(img.shape[:2], dtype = "uint8")
    # ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    res = cv2.merge([img, img, img])
    return res

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

def spiltimage(img):
    h, w = img.shape[:2]
    size = min(h,w)
    if w >= h:
        img1 = img[:,0:size]
        img2 = img[:,w-size:w]
    else:
        img1 = img[0:size,:]
        img2 = img[h-size:h,:]

    return img1,img2

def mergeimage(img1,img2,orgin_image):
    h, w = orgin_image.shape[:2]
    new_img1 = np.zeros((h,w), dtype = "uint8")
    new_img2 = np.zeros((h,w), dtype = "uint8")

    size = min(h,w)
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
