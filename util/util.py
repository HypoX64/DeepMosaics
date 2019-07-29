import os
import shutil
def Traversal(filedir):
    file_list=[]
    for root,dirs,files in os.walk(filedir): 
        for file in files:
            file_list.append(os.path.join(root,file)) 
        for dir in dirs:
            Traversal(dir)
    return file_list

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
        
def  writelog(path,log):
    f = open(path,'a+')
    f.write(log+'\n')

# def del_all(dir_path):
#     files = Traversal(dir_path)
#     for file in files:
#         os.remove(file)
#     os.removedirs(dir_path)

def clean_tempfiles(tmp_init=True):
    if os.path.isdir('./tmp'):   
        shutil.rmtree('./tmp')
    if tmp_init:
        os.makedirs('./tmp')
        os.makedirs('./tmp/video2image')
        os.makedirs('./tmp/addmosaic_image')
        os.makedirs('./tmp/mosaic_crop')
        os.makedirs('./tmp/replace_mosaic')

def file_init(opt):
    if not os.path.isdir(opt.result_dir):
        os.makedirs(opt.result_dir)
        print('makedir:',opt.result_dir)
    clean_tempfiles()
