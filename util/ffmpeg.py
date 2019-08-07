import os,json

def video2image(videopath,imagepath):
    os.system('ffmpeg -i "'+videopath+'" -f image2 '+imagepath)

def video2voice(videopath,voicepath):
    os.system('ffmpeg -i '+videopath+' -f mp3 '+voicepath)

def image2video(fps,imagepath,voicepath,videopath):
    os.system('ffmpeg -y -r '+str(fps)+' -i '+imagepath+' -vcodec libx264 '+'./tmp/video_tmp.mp4')
    #os.system('ffmpeg -f image2 -i '+imagepath+' -vcodec libx264 -r '+str(fps)+' ./tmp/video_tmp.mp4')
    os.system('ffmpeg -i ./tmp/video_tmp.mp4 -i '+voicepath+' -vcodec copy -acodec copy '+videopath)

def get_video_infos(videopath):
    cmd_str =  'ffprobe -v quiet -print_format json -show_format -show_streams -i "' + videopath + '"'  
    out_string = os.popen(cmd_str).read()
    infos = json.loads(out_string)
    fps = eval(infos['streams'][0]['avg_frame_rate'])
    endtime = float(infos['format']['duration'])
    width = int(infos['streams'][0]['width'])
    height = int(infos['streams'][0]['height'])
    return fps,endtime,width,height

