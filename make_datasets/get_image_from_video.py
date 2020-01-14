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
output_dir = './dataset/v2im'
FPS = 1
util.makedirs(output_dir)
for video in videos:
    ffmpeg.continuous_screenshot(video, output_dir, FPS)