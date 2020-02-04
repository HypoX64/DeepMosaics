import argparse
import os
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #base
        self.parser.add_argument('--use_gpu',type=int,default=1, help='if 0 or -1, do not use gpu')
        # self.parser.add_argument('--use_gpu', action='store_true', help='if input it, use gpu')
        self.parser.add_argument('--media_path', type=str, default='./hands_test.mp4',help='your videos or images path')
        self.parser.add_argument('--mode', type=str, default='auto',help='auto | add | clean | style')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_models/add_hands_128.pth',help='pretrained model path')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='output media will be saved here')
        self.parser.add_argument('--tempimage_type', type=str, default='png',help='type of temp image, png | jpg, png is better but occupy more storage space')
        self.parser.add_argument('--netG', type=str, default='auto',
            help='select model to use for netG(Clean mosaic and Transfer style) -> auto | unet_128 | unet_256 | resnet_9blocks | HD | video')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output file,if 0 -> origin')
        
        #AddMosaic
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg | random')
        self.parser.add_argument('--mosaic_size', type=int, default=0,help='mosaic size,if 0 auto size')
        self.parser.add_argument('--mask_extend', type=int, default=10,help='more mosaic area')
        self.parser.add_argument('--mask_threshold', type=int, default=64,help='threshold of recognize mosaic position 0~255')
        
        #CleanMosaic     
        self.parser.add_argument('--mosaic_position_model_path', type=str, default='auto',help='name of model use to find mosaic position')
        self.parser.add_argument('--no_feather', action='store_true', help='if true, no edge feather and color correction, but run faster')
        self.parser.add_argument('--no_large_area', action='store_true', help='if true, do not find the largest mosaic area')
        self.parser.add_argument('--medfilt_num', type=int, default=11,help='medfilt window of mosaic movement in the video')
        self.parser.add_argument('--ex_mult', type=str, default='auto',help='mosaic area expansion')
        
        #StyleTransfer
        self.parser.add_argument('--preprocess', type=str, default='resize', help='resize and cropping of images at load time [ resize | resize_scale_width | edges | gray] or resize,edges(use comma to split)')
        self.parser.add_argument('--edges', action='store_true', help='if true, use edges to generate pictures,(input_nc = 1)')  
        self.parser.add_argument('--canny', type=int, default=150,help='threshold of canny')
        self.parser.add_argument('--only_edges', action='store_true', help='if true, output media will be edges')

        self.initialized = True


    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        model_name = os.path.basename(self.opt.model_path)

        if torch.cuda.is_available() and self.opt.use_gpu > 0:
            self.opt.use_gpu = True
        else:
            self.opt.use_gpu = False


        if self.opt.mode == 'auto':
            if 'add' in model_name:
                self.opt.mode = 'add'
            elif 'clean' in model_name:
                self.opt.mode = 'clean'
            elif 'style' in model_name or 'edges' in model_name:
                self.opt.mode = 'style'
            else:
                print('Please input running mode!')

        if self.opt.output_size == 0 and self.opt.mode == 'style':
            self.opt.output_size = 512

        if 'edges' in model_name or 'edges' in self.opt.preprocess:
            self.opt.edges = True

        if self.opt.netG == 'auto' and self.opt.mode =='clean':
            if 'unet_128' in model_name:
                self.opt.netG = 'unet_128'
            elif 'resnet_9blocks' in model_name:
                self.opt.netG = 'resnet_9blocks'
            elif 'HD' in model_name:
                self.opt.netG = 'HD'
            elif 'video' in model_name:
                self.opt.netG = 'video'
            else:
                print('Type of Generator error!')

        if self.opt.ex_mult == 'auto':
            if 'face' in model_name:
                self.opt.ex_mult = 1.1
            else:
                self.opt.ex_mult = 1.5
        else:
            self.opt.ex_mult = float(self.opt.ex_mult)

        if self.opt.mosaic_position_model_path == 'auto':
            _path = os.path.join(os.path.split(self.opt.model_path)[0],'mosaic_position.pth')
            self.opt.mosaic_position_model_path = _path
            # print(self.opt.mosaic_position_model_path)

        return self.opt