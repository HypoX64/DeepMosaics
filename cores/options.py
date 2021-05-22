import argparse
import os
import sys


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #base
        self.parser.add_argument('--debug', action='store_true', help='if specified, start debug mode')
        self.parser.add_argument('--gpu_id', type=str,default='0', help='if -1, use cpu')
        self.parser.add_argument('--media_path', type=str, default='./imgs/ruoruo.jpg',help='your videos or images path')
        self.parser.add_argument('-ss', '--start_time', type=str, default='00:00:00',help='start position of video, default is the beginning of video')
        self.parser.add_argument('-t', '--last_time', type=str, default='00:00:00',help='duration of the video, default is the entire video')
        self.parser.add_argument('--mode', type=str, default='auto',help='Program running mode. auto | add | clean | style')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_models/mosaic/add_face.pth',help='pretrained model path')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='output media will be saved here')
        self.parser.add_argument('--temp_dir', type=str, default='./tmp', help='Temporary files will go here')
        self.parser.add_argument('--tempimage_type', type=str, default='jpg',help='type of temp image, png | jpg, png is better but occupy more storage space')
        self.parser.add_argument('--netG', type=str, default='auto',
            help='select model to use for netG(Clean mosaic and Transfer style) -> auto | unet_128 | unet_256 | resnet_9blocks | HD | video')
        self.parser.add_argument('--fps', type=int, default=0,help='read and output fps, if 0-> origin')
        self.parser.add_argument('--no_preview', action='store_true', help='if specified,do not preview images when processing video. eg.(when run it on server)')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output media, if 0 -> origin')
        self.parser.add_argument('--mask_threshold', type=int, default=64,help='Mosaic detection threshold (0~255). The smaller is it, the more likely judged as a mosaic area.')

        #AddMosaic
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg | random')
        self.parser.add_argument('--mosaic_size', type=int, default=0,help='mosaic size,if 0 auto size')
        self.parser.add_argument('--mask_extend', type=int, default=10,help='extend mosaic area')
        
        #CleanMosaic     
        self.parser.add_argument('--mosaic_position_model_path', type=str, default='auto',help='name of model use to find mosaic position')
        self.parser.add_argument('--traditional', action='store_true', help='if specified, use traditional image processing methods to clean mosaic')
        self.parser.add_argument('--tr_blur', type=int, default=10, help='ksize of blur when using traditional method, it will affect final quality')
        self.parser.add_argument('--tr_down', type=int, default=10, help='downsample when using traditional method,it will affect final quality')
        self.parser.add_argument('--no_feather', action='store_true', help='if specified, no edge feather and color correction, but run faster')
        self.parser.add_argument('--all_mosaic_area', action='store_true', help='if specified, find all mosaic area, else only find the largest area')
        self.parser.add_argument('--medfilt_num', type=int, default=11,help='medfilt window of mosaic movement in the video')
        self.parser.add_argument('--ex_mult', type=str, default='auto',help='mosaic area expansion')
        
        #StyleTransfer
        self.parser.add_argument('--preprocess', type=str, default='resize', help='resize and cropping of images at load time [ resize | resize_scale_width | edges | gray] or resize,edges(use comma to split)')
        self.parser.add_argument('--edges', action='store_true', help='if specified, use edges to generate pictures,(input_nc = 1)')  
        self.parser.add_argument('--canny', type=int, default=150,help='threshold of canny')
        self.parser.add_argument('--only_edges', action='store_true', help='if specified, output media will be edges')

        self.initialized = True


    def getparse(self, test_flag = False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        model_name = os.path.basename(self.opt.model_path)
        self.opt.temp_dir = os.path.join(self.opt.temp_dir, 'DeepMosaics_temp')

        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)
            import torch
            if not torch.cuda.is_available():
                self.opt.gpu_id = '-1'
        # else:
        #     self.opt.gpu_id = '-1'

        if test_flag:
            if not os.path.exists(self.opt.media_path):
                print('Error: Media does not exist!')
                input('Please press any key to exit.\n')
                sys.exit(0)
            if not os.path.exists(self.opt.model_path):
                print('Error: Model does not exist!')
                input('Please press any key to exit.\n')
                sys.exit(0)

            if self.opt.mode == 'auto':
                if 'clean' in model_name or self.opt.traditional:
                    self.opt.mode = 'clean'
                elif 'add' in model_name:
                    self.opt.mode = 'add'
                elif 'style' in model_name or 'edges' in model_name:
                    self.opt.mode = 'style'
                else:
                    print('Please check model_path!')
                    input('Please press any key to exit.\n')
                    sys.exit(0)

            if self.opt.output_size == 0 and self.opt.mode == 'style':
                self.opt.output_size = 512

            if 'edges' in model_name or 'edges' in self.opt.preprocess:
                self.opt.edges = True

            if self.opt.netG == 'auto' and self.opt.mode =='clean':
                if 'unet_128' in model_name:
                    self.opt.netG = 'unet_128'
                elif 'resnet_9blocks' in model_name:
                    self.opt.netG = 'resnet_9blocks'
                elif 'HD' in model_name and 'video' not in model_name:
                    self.opt.netG = 'HD'
                elif 'video' in model_name:
                    self.opt.netG = 'video'
                else:
                    print('Type of Generator error!')
                    input('Please press any key to exit.\n')
                    sys.exit(0)

            if self.opt.ex_mult == 'auto':
                if 'face' in model_name:
                    self.opt.ex_mult = 1.1
                else:
                    self.opt.ex_mult = 1.5
            else:
                self.opt.ex_mult = float(self.opt.ex_mult)

            if self.opt.mosaic_position_model_path == 'auto' and self.opt.mode == 'clean':
                _path = os.path.join(os.path.split(self.opt.model_path)[0],'mosaic_position.pth')
                if os.path.isfile(_path):
                    self.opt.mosaic_position_model_path = _path
                else:
                    input('Please check mosaic_position_model_path!')
                    input('Please press any key to exit.\n')
                    sys.exit(0)

        return self.opt