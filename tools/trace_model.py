import os
import sys
import traceback
sys.path.append("..")
from util import mosaic
import torch

try:
    from cores import Options,add,clean,style
    from util import util
    from models import loadmodel
except Exception as e:
    print(e)
    input('Please press any key to exit.\n')
    sys.exit(0)

opt = Options().getparse(test_flag = False)
if not os.path.isdir(opt.temp_dir):
    util.file_init(opt)

def saveScriptModel(model,example,savepath):
    model.cpu()
    traced_script_module = torch.jit.trace(model, example)
    # try ScriptModel
    output = traced_script_module(example)
    print(output)
    traced_script_module.save(savepath)

savedir = '../cpp/res/models/'
util.makedirs(savedir)

opt.mosaic_position_model_path = '../pretrained_models/mosaic/mosaic_position.pth'
model = loadmodel.bisenet(opt,'mosaic')
example = torch.ones((1,3,360,360))
saveScriptModel(model,example,os.path.join(savedir,'mosaic_position.pt'))



# def main():
    
#     if os.path.isdir(opt.media_path):
#         files = util.Traversal(opt.media_path)
#     else:
#         files = [opt.media_path]        
#     if opt.mode == 'add':
#         netS = loadmodel.bisenet(opt,'roi')
#         for file in files:
#             opt.media_path = file
#             if util.is_img(file):
#                 add.addmosaic_img(opt,netS)
#             elif util.is_video(file):
#                 add.addmosaic_video(opt,netS)
#                 util.clean_tempfiles(opt, tmp_init = False)
#             else:
#                 print('This type of file is not supported')
#             util.clean_tempfiles(opt, tmp_init = False)

#     elif opt.mode == 'clean':
#         netM = loadmodel.bisenet(opt,'mosaic')
#         if opt.traditional:
#             netG = None
#         elif opt.netG == 'video':
#             netG = loadmodel.video(opt)
#         else:
#             netG = loadmodel.pix2pix(opt)
        
#         for file in files:
#             opt.media_path = file
#             if util.is_img(file):
#                 clean.cleanmosaic_img(opt,netG,netM)
#             elif util.is_video(file):
#                 if opt.netG == 'video' and not opt.traditional:            
#                     clean.cleanmosaic_video_fusion(opt,netG,netM)
#                 else:
#                     clean.cleanmosaic_video_byframe(opt,netG,netM)
#                 util.clean_tempfiles(opt, tmp_init = False)
#             else:
#                 print('This type of file is not supported')

#     elif opt.mode == 'style':
#         netG = loadmodel.style(opt)
#         for file in files:
#             opt.media_path = file
#             if util.is_img(file):
#                 style.styletransfer_img(opt,netG)
#             elif util.is_video(file):
#                 style.styletransfer_video(opt,netG)
#                 util.clean_tempfiles(opt, tmp_init = False)
#             else:
#                 print('This type of file is not supported')

#     util.clean_tempfiles(opt, tmp_init = False)

# if __name__ == '__main__':
#     main()