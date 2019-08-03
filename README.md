![image](https://github.com/HypoX64/DeepMosaics/blob/master/imgs/hand.gif)
# DeepMosaics
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>
This porject based on ‘semantic segmentation’ and ‘Image-to-Image Translation’.
<br>

## Notice
The code do not include the part of training, I will finish it in my free time.
<br>

## Run DeepMosaics
You can either run DeepMosaics via pre-built binary package or from source.<br>

### Pre-built binary package
For windows, we bulid a GUI version for easy test.<br>
Download this version via [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>

![image](https://github.com/HypoX64/DeepMosaics/blob/master/imgs/GUI.png)<br>

Attentions:<br>
- Require Windows_x86_64, Windows10 is better.<br>
- File path cannot contain spaces (" ").<br>
- Run time depends on computer performance.<br>
- If output video cannot be played, you can try it with [potplayer](https://daumpotplayer.com/download/).

### Run from source
#### Prerequisites
- Linux, Mac OS, Windows
- Python 3.6+
- [ffmpeg 3.4](http://ffmpeg.org/)
- [Pytorch 1.0+](https://pytorch.org/)  [(Old version codes)](https://github.com/HypoX64/DeepMosaics/tree/Pytorch0.4)
- CPU or NVIDIA GPU + CUDA CuDNN<br>
#### Dependencies
This code depends on opencv-python, torchvision available via pip install.
#### Clone this repo:
```bash
git clone https://github.com/HypoX64/DeepMosaics
cd DeepMosaics
```
#### Get pre_trained models and test video
You can download pre_trained models and test video and replace the files in the project.<br>
[[Google Drive]](https://drive.google.com/open?id=10nARsiZoZGcaKw40nQu9fJuRp1oeabPs)   [[百度云,提取码7thu]](https://pan.baidu.com/s/1IG4bdIiIC9PH9-oEyae5Sg) 

#### Simple example
* Add Mosaic (output video will save in './result')
```bash
python3 deepmosaic.py
```
* Clean Mosaic (output video will save in './result')
```bash
python3 deepmosaic.py --mode clean --model_path ./pretrained_models/clean_hands_unet_128.pth --media_path ./result/hands_test_AddMosaic.mp4
```
#### More parameters
If you want to test other image or video, please refer to this file.
[[options.py]](https://github.com/HypoX64/DeepMosaics/blob/master/options.py) 
<br>

## Acknowledgments
This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet).
