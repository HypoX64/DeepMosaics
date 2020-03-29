import os
import shutil

def findalldir(rootdir):
    dir_list = []
    for root,dirs,files in os.walk(rootdir): 
        for dir in dirs:
            dir_list.append(os.path.join(root,dir))
    return(dir_list)

def Traversal(filedir):
    file_list=[]
    dir_list = []
    for root,dirs,files in os.walk(filedir): 
        for file in files:
            file_list.append(os.path.join(root,file)) 
        for dir in dirs:
            dir_list.append(os.path.join(root,dir))
            Traversal(dir)
    return file_list,dir_list

def is_img(path):
    ext = os.path.splitext(path)[1]
    ext = ext.lower()
    if ext in ['.jpg','.png','.jpeg','.bmp']:
        return True
    else:
        return False

def is_video(path):
    ext = os.path.splitext(path)[1]
    ext = ext.lower()
    if ext in ['.mp4','.flv','.avi','.mov','.mkv','.wmv','.rmvb']:
        return True
    else:
        return False

def cleanall():
    file_list,dir_list = Traversal('./')
    for file in file_list:
        if ('tmp' in file) | ('pth' in file)|('pycache' in file) | is_video(file) | is_img(file):
            if os.path.exists(file):
                if 'imgs' not in file:
                    os.remove(file)
                    print('remove file:',file)

    for dir in dir_list:
        if ('tmp'in dir)|('pycache'in dir):
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print('remove dir:',dir)