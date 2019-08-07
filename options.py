import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #base
        self.parser.add_argument('--use_gpu', action='store_true', help='if input it, use gpu')
        self.parser.add_argument('--media_path', type=str, default='./hands_test.mp4',help='your videos or images path')
        self.parser.add_argument('--mode', type=str, default='add',help='add or clean mosaic into your media  add | clean')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_models/add_hands_128.pth',help='pretrained model path')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='output result will be saved here')
        self.parser.add_argument('--tempimage_type', type=str, default='png',help='type of temp image, png | jpg, png is better but occupy more storage space')

        #AddMosaic
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg | random')
        self.parser.add_argument('--mosaic_size', type=int, default=0,help='mosaic size,if 0 auto size')
        self.parser.add_argument('--mask_extend', type=int, default=10,help='more mosaic area')
        self.parser.add_argument('--mask_threshold', type=int, default=64,help='threshold of recognize mosaic position 0~255')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output file,if 0 -> origin')
        
        #AddMosaic
        self.parser.add_argument('--netG', type=str, default='auto',help='select model to use for netG(clean mosaic) -> auto | unet_128 | resnet_9blocks')
        self.parser.add_argument('--mosaic_position_model_path', type=str, default='auto',help='name of model use to find mosaic position')
        self.parser.add_argument('--no_feather', action='store_true', help='if true, no edge feather,but run faster')
        self.parser.add_argument('--medfilt_num', type=int, default=11,help='medfilt window of mosaic movement in the video')
        self.initialized = True


    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.netG == 'auto':
            if 'unet_128' in self.opt.model_path:
                self.opt.netG = 'unet_128'
            elif 'resnet_9blocks' in self.opt.model_path:
                self.opt.netG = 'resnet_9blocks'

        if self.opt.mosaic_position_model_path == 'auto':
            _path = os.path.join(os.path.split(self.opt.model_path)[0],'mosaic_position.pth')
            self.opt.mosaic_position_model_path = _path
            # print(self.opt.mosaic_position_model_path)

        return self.opt