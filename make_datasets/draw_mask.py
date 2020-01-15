import cv2
import numpy as np
import datetime
import os
import random

import sys
sys.path.append("..")
from util import util
from util import image_processing as impro

image_dir = './datasets_img/v2im'
mask_dir = './datasets_img/v2im_mask'
util.makedirs(mask_dir)

files = os.listdir(image_dir)
files_new =files.copy()
print('find image:',len(files))
masks = os.listdir(mask_dir)
print('mask:',len(masks))

# mouse callback function
drawing = False # true if mouse is pressed
ix,iy = -1,-1
brushsize = 20
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,brushsize

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),brushsize,(0,255,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),brushsize,(0,255,0),-1)

def makemask(img):
    # starttime = datetime.datetime.now()
    mask = np.zeros(img.shape, np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # if (img[row,col,:] == [0,255,0]).all(): #too slow
            if img[row,col,0] == 0:
                if img[row,col,1] == 255:
                    if img[row,col,2] == 0:
                        mask[row,col,:] = [255,255,255]
    # endtime = datetime.datetime.now()
    # print('Cost time:',(endtime-starttime))
    return mask


for i in range(len(masks)):
    masks[i]=masks[i].replace('.png','.jpg')
for file in files:
    if file  in masks:
        files_new.remove(file)
files = files_new
# files = list(set(files)) #Distinct 
print('remain:',len(files))
random.shuffle(files)
# files.sort()
cnt = 0

for file in files:
    cnt += 1
    img = cv2.imread(os.path.join(image_dir,file))
    img = impro.resize(img,512)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle) #MouseCallback
    while(1):

        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            img = impro.resize(img,256)
            mask = makemask(img)
            cv2.imwrite(os.path.join(mask_dir,os.path.splitext(file)[0]+'.png'),mask)
            print(os.path.join(mask_dir,os.path.splitext(file)[0]+'.png'))
            # cv2.destroyAllWindows()
            print('remain:',len(files)-cnt)
            brushsize = 20
            break
        elif k == ord('a'):
            brushsize -= 5
            if brushsize<5:
                brushsize = 5
            print('brushsize:',brushsize)
        elif k == ord('d'):
            brushsize += 5
            print('brushsize:',brushsize)
        elif k == ord('w'):
            print('remain:',len(files)-cnt)
            break

# cv2.destroyAllWindows()