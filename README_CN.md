![image](./imgs/hand.gif)
# <img src="./imgs/icon.jpg" width="48">DeepMosaics
这是一个通过深度学习自动的为图片/视频添加马赛克,或消除马赛克的项目.<br>它基于“语义分割”以及“图像翻译”.<br>

### 更多例子

原始 | 自动打码 |  自动去码  
:-:|:-:|:-:
![image](./imgs/example/lena.jpg) | ![image](./imgs/example/lena_add.jpg) | ![image](./imgs/example/lena_clean.jpg) 
![image](./imgs/example/youknow.png)  | ![image](./imgs/example/youknow_add.png) | ![image](./imgs/example/youknow_clean.png) 

* 与 [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)相比较

马赛克图片 | DeepCreamPy | ours  
:-:|:-:|:-:
![image](./imgs/example/face_a_mosaic.jpg) | ![image](./imgs/example/a_dcp.png) | ![image](./imgs/example/face_a_clean.jpg) 
![image](./imgs/example/face_b_mosaic.jpg) | ![image](./imgs/example/b_dcp.png) | ![image](./imgs/example/face_b_clean.jpg) 

* 风格转换

原始 | 梵高风格 | 转化为冬天
:-:|:-:|:-:
![image](./imgs/example/SZU.jpg) | ![image](./imgs/example/SZU_vangogh.jpg) | ![image](./imgs/example/SZU_summer2winter.jpg) 

一个有意思的尝试:[香蕉君♂猫](https://www.bilibili.com/video/BV1Q7411W7n6)

## 如何运行
可以通过我们预编译好的二进制包或源代码运行.<br>

### 预编译的程序包
对于Windows用户,我们提供了包含GUI界面的免安装软件包.<br>
可以通过下面两种方式进行下载: [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>

* [[使用教程]](./docs/exe_help_CN.md)<br>

![image](./imgs/GUI.png)<br>

注意事项:<br>
  - 程序的运行要求在64位Windows操作系统,我仅在Windows10运行过,其他版本暂未经过测试<br>
  - 请根据需求选择合适的预训练模型进行测试，不同的预期训练模型具有不同的效果.[[预训练模型介绍]](./docs/pre-trained_models_introduction_CN.md)<br>
  - 运行时间取决于电脑性能,对于视频文件,我们建议使用源码并在GPU上运行.<br>
  - 如果输出的视频无法播放,这边建议您尝试[potplayer](https://daumpotplayer.com/download/).<br>
  - 相比于源码,该版本的更新将会延后.

### 通过源代码运行
#### 前提要求
  - Linux, Mac OS, Windows
  - Python 3.6+
  - [ffmpeg 3.4.6](http://ffmpeg.org/)
  - [Pytorch 1.0+](https://pytorch.org/) 
  - CPU or NVIDIA GPU + CUDA CuDNN<br>
#### Python依赖项
代码依赖于opencv-python以及 torchvision,可有通过pip install 进行安装.
#### 克隆源代码
```bash
git clone https://github.com/HypoX64/DeepMosaics
cd DeepMosaics
```
#### 下载预训练模型
可以通过以下两种方法下载预训练模型,并将他们置于'./pretrained_models'文件夹中.<br>
[[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>
[[预训练模型介绍]](./docs/pre-trained_models_introduction_CN.md)<br>

#### 简单的例子
* 为视频添加马赛克,例子中认为脸是需要打码的区域 ,可以通过切换预训练模型切换自动打码区域(输出结果将储存到 './result')<br>
```bash
python deepmosaic.py --media_path ./imgs/ruoruo.jpg --model_path ./pretrained_models/mosaic/add_face.pth --use_gpu 0
```
* 将视频中的马赛克移除,对于不同的打码物体需要使用对应的预训练模型进行马赛克消除(输出结果将储存到  './result')<br>
```bash
python deepmosaic.py --media_path ./result/ruoruo_add.jpg --model_path ./pretrained_models/mosaic/clean_face_HD.pth --use_gpu 0
```
#### 更多的参数
如果想要测试其他的图片或视频,请参照以下文件输入参数.<br>
[[options_introduction_CN.md]](./docs/options_introduction_CN.md) <br>

## 使用自己的数据训练模型
如果需要使用自己的数据训练模型，请参照 [training_with_your_own_dataset.md](./docs/training_with_your_own_dataset.md)

## 鸣谢
代码大量的参考了以下项目:[[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet) [[pix2pixHD]](https://github.com/NVIDIA/pix2pixHD) [[BiSeNet]](https://github.com/ooooverflow/BiSeNet).

