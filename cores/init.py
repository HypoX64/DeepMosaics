import os
from util import util,ffmpeg

'''
---------------------Video Init---------------------
'''
def video_init(opt,path):
    fps,endtime,height,width = ffmpeg.get_video_infos(path)
    if opt.fps !=0:
        fps = opt.fps

    # resume
    if os.path.isfile(os.path.join(opt.temp_dir,'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir,'step.json'))
        if int(step['step'])>=1:
            choose = input('There is an unfinished video. Continue it? [y/n] ')
            if choose.lower() =='yes' or choose.lower() == 'y':
                imagepaths = os.listdir(opt.temp_dir+'/video2image')
                imagepaths.sort()
                return fps,imagepaths,height,width
    
    print('Step:1/4 -- Convert video to images')
    util.file_init(opt)
    ffmpeg.video2voice(path,opt.temp_dir+'/voice_tmp.mp3',opt.start_time,opt.last_time)
    ffmpeg.video2image(path,opt.temp_dir+'/video2image/output_%06d.'+opt.tempimage_type,fps,opt.start_time,opt.last_time)
    imagepaths = os.listdir(opt.temp_dir+'/video2image')
    imagepaths.sort()
    step = {'step':2,'frame':0}
    util.savejson(os.path.join(opt.temp_dir,'step.json'),step)

    return fps,imagepaths,height,width