import os
from cores import Options,core
from util import util

opt = Options().getparse()
util.file_init(opt)

def main():
    
    if opt.mode == 'add':
        if util.is_img(opt.media_path):
            core.addmosaic_img(opt)
        elif util.is_video(opt.media_path):
            core.addmosaic_video(opt)
        else:
            print('This type of file is not supported')

    elif opt.mode == 'clean':
        if util.is_img(opt.media_path):
            core.cleanmosaic_img(opt)
        elif util.is_video(opt.media_path):
            if opt.netG == 'video':
                core.cleanmosaic_video_fusion(opt)
            else:
                core.cleanmosaic_video_byframe(opt)
        else:
            print('This type of file is not supported')

    util.clean_tempfiles(tmp_init = False)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:',e)
        input('Please press any key to exit.\n')
        util.clean_tempfiles(tmp_init = False)
        exit(0)
