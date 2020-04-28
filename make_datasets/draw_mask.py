import cv2
import numpy as np
import datetime
import os
import random

import sys
sys.path.append("..")
from cores import Options
from util import util
from util import image_processing as impro


opt = Options()
opt.parser.add_argument('--datadir',type=str,default=' ', help='your images dir')
opt.parser.add_argument('--savedir',type=str,default='../datasets/draw/face', help='')
opt = opt.getparse()

mask_savedir = os.path.join(opt.savedir,'mask')
img_savedir = os.path.join(opt.savedir,'origin_image')
util.makedirs(mask_savedir)
util.makedirs(img_savedir)

filepaths = util.Traversal(opt.datadir)
filepaths = util.is_imgs(filepaths)
random.shuffle(filepaths)
print('find image:',len(filepaths))

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
            cv2.circle(img_drawn,(x,y),brushsize,(0,255,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img_drawn,(x,y),brushsize,(0,255,0),-1)

def makemask(img_drawn):
    # starttime = datetime.datetime.now()
    mask = np.zeros(img_drawn.shape, np.uint8)
    for row in range(img_drawn.shape[0]):
        for col in range(img_drawn.shape[1]):
            # if (img_drawn[row,col,:] == [0,255,0]).all(): #too slow
            if img_drawn[row,col,0] == 0:
                if img_drawn[row,col,1] == 255:
                    if img_drawn[row,col,2] == 0:
                        mask[row,col,:] = [255,255,255]
    return mask

cnt = 0
for file in filepaths:
    try:
        cnt += 1
        img = impro.imread(file,loadsize=512)
        img_drawn = img.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle) #MouseCallback
        while(1):

            cv2.imshow('image',img_drawn)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                
                img_drawn = impro.resize(img_drawn,256)
                mask = makemask(img_drawn)
                cv2.imwrite(os.path.join(mask_savedir,os.path.splitext(os.path.basename(file))[0]+'.png'),mask)
                cv2.imwrite(os.path.join(img_savedir,os.path.basename(file)),img)   
                print('Saved:',os.path.join(mask_savedir,os.path.splitext(os.path.basename(file))[0]+'.png'),mask)
                # cv2.destroyAllWindows()
                print('remain:',len(filepaths)-cnt)
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
                print('remain:',len(filepaths)-cnt)
                break
    except Exception as e:
        print(file,e)

