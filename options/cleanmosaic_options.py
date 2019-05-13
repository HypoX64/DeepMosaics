import argparse
import os

class CleanOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--use_gpu', action='store_true', help='if true, use gpu')
        self.parser.add_argument('--input_dir', type=str, default='./video_or_image',help='put your videos or images here')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='result will be saved here')
        self.parser.add_argument('--model_dir', type=str, default='./pretrained_models/CleanMosaic',
                                help='put pre_train model here, including 1.model use to find mosaic position 2.model use to clean mosaic')
        self.parser.add_argument('--model_name', type=str, default='hands_unet_128.pth',help='name of model use to clean mosaic')
        self.parser.add_argument('--model_type_netG', type=str, default='unet_128',help='select model to use for netG')
        self.parser.add_argument('--mosaic_position_model_name', type=str, default='mosaic_position.pth',
                                help='name of model use to find mosaic position')
        self.parser.add_argument('--no_feather', action='store_true', help='if true, no edge feather,but run faster')
        self.parser.add_argument('--medfilt_num', type=int, default=11,help='medfilt window of mosaic movement in the video')
        self.parser.add_argument('--tempimage_type', type=str, default='png',help='type of temp image, png | jpg, png is better but occupy more storage space')
#       self.parser.add_argument('--zoom_multiple', type=float, default=1.0,help='zoom video')
        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt