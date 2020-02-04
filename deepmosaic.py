import os
from cores import Options,core
from util import util
from models import loadmodel

opt = Options().getparse()
util.file_init(opt)

def main():
    
    if os.path.isdir(opt.media_path):
        files = util.Traversal(opt.media_path)
    else:
        files = [opt.media_path]        
    if opt.mode == 'add':
        netS = loadmodel.unet(opt)
        for file in files:
            opt.media_path = file
            if util.is_img(file):
                core.addmosaic_img(opt,netS)
            elif util.is_video(file):
                core.addmosaic_video(opt,netS)
            else:
                print('This type of file is not supported')

    elif opt.mode == 'clean':
        netM = loadmodel.unet_clean(opt)
        if opt.netG == 'video':
            netG = loadmodel.video(opt)
        else:
            netG = loadmodel.pix2pix(opt)
        
        for file in files:
            opt.media_path = file
            if util.is_img(file):
                core.cleanmosaic_img(opt,netG,netM)
            elif util.is_video(file):
                if opt.netG == 'video':            
                    core.cleanmosaic_video_fusion(opt,netG,netM)
                else:
                    core.cleanmosaic_video_byframe(opt,netG,netM)
            else:
                print('This type of file is not supported')

    elif opt.mode == 'style':
        netG = loadmodel.style(opt)
        for file in files:
            opt.media_path = file
            if util.is_img(file):
                core.styletransfer_img(opt,netG)
            elif util.is_video(file):
                core.styletransfer_video(opt,netG)

            else:
                print('This type of file is not supported')

    util.clean_tempfiles(tmp_init = False)
        
# main()
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:',e)
        input('Please press any key to exit.\n')
        util.clean_tempfiles(tmp_init = False)
        exit(0)
