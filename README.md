![image](https://github.com/HypoX64/DeepMosaics/blob/master/hand.gif)
# DeepMosaics
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>
This porject based on semantic segmentation and pix2pix.
<br>

## Notes
The code do not include the part of training, I will finish it in my free time.
<br>

## Prerequisites
- Linux, (I didn't try this code on Windows or Mac OS)
- Python 3.6+
- ffmpeg
- Pytorch 1.0+  [(Old version codes)](https://github.com/HypoX64/DeepMosaics/tree/Pytorch0.4)
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Clone this repo:
```bash
git clone https://github.com/HypoX64/DeepMosaics
cd DeepMosaics
```
### Get pre_trained models and test video
You can download pre_trained models and test video and replace the files in the project.<br>
[[Google Drive]](https://drive.google.com/open?id=10nARsiZoZGcaKw40nQu9fJuRp1oeabPs)
 [[百度云,提取码7thu]](https://pan.baidu.com/s/1IG4bdIiIC9PH9-oEyae5Sg) 

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
