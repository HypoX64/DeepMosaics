![image](https://github.com/HypoX64/DeepMosaics/blob/master/hand.gif)
# DeepMosaics
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>
This porject based on semantic segmentation and pix2pix.
<br>

## Notice
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
This code depends on opencv-python, available via pip install.
### Simple example
* Add Mosaic (output video will save in './result')
```bash
python3 deepmosaic.py
```
* Clean Mosaic (output video will save in './result')
```bash
python3 deepmosaic.py --mode clean --model_path ./pretrained_models/clean_hands_unet_128.pth --media_path ./result/hands_test_AddMosaic.mp4
```
### More parameters
If you want to test other image or video, please refer to this file.
[[options.py]](https://github.com/HypoX64/DeepMosaics/blob/master/options.py) 
<br>

## Acknowledgments
This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet).
