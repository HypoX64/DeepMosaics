![image](https://github.com/HypoX64/DeepMosaics/blob/master/hand.gif)
# DeepMosaics
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>
This porject based on semantic segmentation and pix2pix.
<br>
## Notes
The code do not include the part of training, I will finish it in my free time.
<br>
## Prerequisites
- Linux, (I didn't try this code on Windows or mac machine)
- Python 3.5+
- ffmpeg
- Pytroch 0.4, (I will update to 1.0)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Clone this repo:
```bash
git clone https://github.com/HypoX64/DeepMosaics
cd DeepMosaics
```
### Get pre_trained models and test video
You can download pre_trained models and test video and replace the files in the project.<br>
[[Google Drive]](https://drive.google.com/open?id=1PXt3dE9Eez2xUqpemLJutwTCC0tW-D2g)
 [[百度云,提取码z8vz]](https://pan.baidu.com/s/1Wi8T6PE4ExTjrHVQhv3rJA) 
### Dependencies
This code depends on numpy, scipy, opencv-python, torchvision, available via pip install.
### AddMosaic
```bash
python3 AddMosaic.py
```
### CleanMosaic
copy the AddMosaic video from './result' to './video_or_image'
```bash
python3 CleanMosaic.py
```
### More parameters
[[addmosaic_options]](https://github.com/HypoX64/DeepMosaics/blob/master/options/addmosaic_options.py)  [[cleanmosaic_options]](https://github.com/HypoX64/DeepMosaics/blob/master/options/cleanmosaic_options.py)
<br>
## Acknowledgments
This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet).
