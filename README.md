![image](./imgs/hand.gif)
# <img src="./imgs/icon.jpg" width="48">DeepMosaics
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>
This porject based on "semantic segmentation" and "Image-to-Image Translation".<br>

* [中文版README](./README_CN.md)<br>

### More example
origin | auto add mosaic |  auto clean mosaic  
:-:|:-:|:-:
![image](./imgs/example/lena.jpg) | ![image](./imgs/example/lena_add.jpg) | ![image](./imgs/example/lena_clean.jpg) 
![image](./imgs/example/youknow.png)  | ![image](./imgs/example/youknow_add.png) | ![image](./imgs/example/youknow_clean.png) 
* Compared with [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)
mosaic image | DeepCreamPy | ours  
:-:|:-:|:-:
![image](./imgs/example/face_a_mosaic.jpg) | ![image](./imgs/example/a_dcp.png) | ![image](./imgs/example/face_a_clean.jpg) 
![image](./imgs/example/face_b_mosaic.jpg) | ![image](./imgs/example/b_dcp.png) | ![image](./imgs/example/face_b_clean.jpg) 
* Style Transfer
origin | to Van Gogh | to winter
:-:|:-:|:-:
![image](./imgs/example/SZU.jpg) | ![image](./imgs/example/SZU_vangogh.jpg) | ![image](./imgs/example/SZU_summer2winter.jpg) 
An interesting example:[Ricardo Milos to cat](https://www.bilibili.com/video/BV1Q7411W7n6)

## Run DeepMosaics
You can either run DeepMosaics via pre-built binary package or from source.<br>

### Pre-built binary package
For windows, we bulid a GUI version for easy test.<br>
Download this version and pre-trained model via [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>

* [[How to use]](./docs/exe_help.md)<br>

![image](./imgs/GUI.png)<br>
Attentions:<br>

  - Require Windows_x86_64, Windows10 is better.<br>
  - Different pre-trained models are suitable for different effects.[[Introduction to pre-trained models]](./docs/pre-trained_models_introduction.md)<br>
  - Run time depends on computer performance(The current version does not support gpu, if you need to use gpu please run source).<br>
  - If output video cannot be played, you can try with [potplayer](https://daumpotplayer.com/download/).<br>
  - GUI version update slower than source.<br>

### Run from source
#### Prerequisites
  - Linux, Mac OS, Windows
  - Python 3.6+
  - [ffmpeg 3.4.6](http://ffmpeg.org/)
  - [Pytorch 1.0+](https://pytorch.org/)
  - CPU or NVIDIA GPU + CUDA CuDNN<br>
#### Dependencies
This code depends on opencv-python, torchvision available via pip install.
#### Clone this repo
```bash
git clone https://github.com/HypoX64/DeepMosaics
cd DeepMosaics
```
#### Get pre-trained models
You can download pre_trained models and put them into './pretrained_models'.<br>
[[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ)<br>
[[Introduction to pre-trained models]](./docs/pre-trained_models_introduction.md)<br>

#### Simple example
* Add Mosaic (output media will save in './result')<br>
```bash
python3 deepmosaic.py --media_path ./imgs/ruoruo.jpg --model_path ./pretrained_models/mosaic/add_face.pth --use_gpu -1
```
* Clean Mosaic (output media will save in './result')<br>
```bash
python3 deepmosaic.py --media_path ./result/ruoruo_add.jpg --model_path ./pretrained_models/mosaic/clean_face_HD.pth --use_gpu -1
```
#### More parameters
If you want to test other image or video, please refer to this file.<br>
[[options_introduction.md]](./docs/options_introduction.md) <br>

## Training with your own dataset
If you want to train with your own dataset, please refer to [training_with_your_own_dataset.md](./docs/training_with_your_own_dataset.md)

## Acknowledgments
This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet) [[pix2pixHD]](https://github.com/NVIDIA/pix2pixHD) [[BiSeNet]](https://github.com/ooooverflow/BiSeNet).

