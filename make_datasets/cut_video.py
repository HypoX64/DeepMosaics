import os
import numpy as np
import cv2
import random
import csv

import sys
sys.path.append("..")
from util import util,ffmpeg
from util import image_processing as impro

files = util.Traversal('/media/hypo/Media/download')
videos = util.is_videos(files)



useable_videos = []
video_dict = {}
reader = csv.reader(open('./csv/video_used_time.csv'))
for line in reader:
    useable_videos.append(line[0])
    video_dict[line[0]]=line[1:]

in_cnt = 0
out_cnt = 1
for video in videos:
    if os.path.basename(video) in useable_videos:

        for i in range(len(video_dict[os.path.basename(video)])):
            ffmpeg.cut_video(video, video_dict[os.path.basename(video)][i], '00:00:05', './video/'+'%04d'%out_cnt+'.mp4')
            out_cnt +=1
        in_cnt += 1          
