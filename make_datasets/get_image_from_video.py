import os
import sys
sys.path.append("..")
from cores import Options
from util import util,ffmpeg

opt = Options()
opt.parser.add_argument('--datadir',type=str,default='', help='your video dir')
opt.parser.add_argument('--savedir',type=str,default='../datasets/video2image', help='')
opt = opt.getparse()

files = util.Traversal(opt.datadir)
videos = util.is_videos(files)

util.makedirs(opt.savedir)
for video in videos:
    ffmpeg.continuous_screenshot(video, opt.savedir, opt.fps)