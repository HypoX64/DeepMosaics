import argparse
import os


class AddOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--use_gpu', action='store_true', help='if true, use gpu')
        self.parser.add_argument('--input_dir', type=str, default='./video_or_image',help='put your videos or images here')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='result will be saved here')
        self.parser.add_argument('--model_dir', type=str, default='./pretrained_models/AddMosaic',
                                help='put pre_train model here')
        self.parser.add_argument('--model_name', type=str, default='hands_128.pth',help='name of model use to Add mosaic')
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg')
        self.parser.add_argument('--mosaic_size', type=int, default=20,help='mosaic size')
        self.parser.add_argument('--mask_extend', type=int, default=20,help='more mosaic')
        self.parser.add_argument('--mask_threshold', type=int, default=64,help='threshold of recognize mosaic position 0~255')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output file,if 0 -> origin')
        
        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt