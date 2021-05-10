## DeepMosaics.exe  使用说明
下载程序以及预训练模型 [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>
[视频教程](https://www.bilibili.com/video/BV1QK4y1a7Av)<br>

注意事项:<br>


  - 程序的运行要求在64位Windows操作系统,我们仅在Windows10运行过,其他版本暂未经过测试<br>
  - 请根据需求选择合适的预训练模型进行测试<br>
  - 运行时间取决于电脑性能,对于视频文件,我们建议使用GPU运行<br>
  - 如果输出的视频无法播放,这边建议您尝试[potplayer](https://daumpotplayer.com/download/).<br>
  - 相比于源码,该版本的更新将会延后.

### 如何安装
#### CPU version
* 1.下载安装 Microsoft Visual C++
  https://aka.ms/vs/16/release/vc_redist.x64.exe
#### GPU version
仅支持gtx1060及以上的NVidia显卡(要求460版本以上的驱动以及11.0版本的CUDA, 注意只能是11.0)
* 1.Download and install Microsoft Visual C++
  https://aka.ms/vs/16/release/vc_redist.x64.exe
* 2.Update your gpu drive to 460(or above)
  https://www.nvidia.com/en-us/geforce/drivers/
* 3.Download and install CUDA 11.0:
  https://developer.nvidia.com/cuda-toolkit-archive

当然这些也能在百度云上下载
https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ
提取码: 1x0a

### 如何使用

* step 1: 选择需要处理的图片或视频
* step 2: 选择预训练模型(不同的预训练模型有不同的效果)
* step 3: 运行程序并等待
* step 4: 查看结果(储存在result文件夹下)

## 预训练模型说明
当前的预训练模型分为两类——添加/移除马赛克以及风格转换.

* 马赛克

|              文件名              |                     描述                      |
| :------------------------------: | :-------------------------------------------: |
|           add_face.pth           |           对图片或视频中的脸部打码            |
|        clean_face_HD.pth         | 对图片或视频中的脸部去码<br>(要求内存 > 8GB). |
|         add_youknow.pth          |        对图片或视频中的...内容打码         |
| clean_youknow_resnet_9blocks.pth |        对图片或视频中的...内容去码         |
|     clean_youknow_video.pth      |           对视频中的...内容去码,推荐使用带有'video'的模型去除视频中的马赛克            |


* 风格转换

|          文件名        |                        描述                        |
| :---------------------: | :-------------------------------------------------------: |
| style_apple2orange.pth  | 苹果变橙子 |
| style_orange2apple.pth  | 橙子变苹果 |
| style_summer2winter.pth |     夏天变冬天     |
| style_winter2summer.pth | 冬天变夏天 |
|    style_cezanne.pth    |            转化为Paul Cézanne 的绘画风格            |
|     style_monet.pth     | 转化为Claude Monet的绘画风格 |
|     style_ukiyoe.pth     | 转化为Ukiyoe的绘画风格 |
|     style_vangogh.pth     | 转化为Van Gogh的绘画风格 |

### GUI界面注释
![image](../imgs/GUI_Instructions.jpg)<br>
* 1. 选择需要处理的图片或视频
* 2. 选择预训练模型
* 3. 程序运行模式  (auto | add | clean | style)
* 4. 使用GPU (该版本目前不支持GPU,若需要使用GPU请使用源码运行).
* 5. 限制输出的视频帧率(0->原始帧率).
* 6. 更多的选项以及参数
* 7. 自行输入更多参数，详见下文
* 8. 运行
* 9. 打开帮助文件
* 10. 支持我们
* 11. 版本信息
* 12. 打开项目的github页面

### 参数说明
如果需要更多的效果,  请按照 '--option your-parameters' 输入所需要的参数
* 基本

|    选项    |        描述         |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
|  --gpu_id   |   if -1, do not use gpu    |                    0                    |
| --media_path | 需要处理的视频或者照片的路径 |            ./imgs/ruoruo.jpg            |
|    --mode    |    运行模式(auto/clean/add/style)    |                 'auto'                  |
| --model_path |   预训练模型的路径    | ./pretrained_models/mosaic/add_face.pth |
| --result_dir | 保存路径 |                 ./result          |
|    --fps    |    限制视频输出的fps，0则为默认    |                 0                  |
*  添加马赛克

|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --mosaic_mod | 马赛克类型 -> squa_avg/ squa_random/ squa_avg_circle_edge/ rect_avg/random |                    squa_avg                    |
| --mosaic_size | 马赛克大小，０则为自动 |            0            |
|    --mask_extend    |    拓展马赛克区域    |         10  |
| --mask_threshold | 马赛克区域识别阈值 0~255,越小越容易被判断为马赛克区域 | 64 |

* 去除马赛克

|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --traditional | 如果输入这个参数则使用传统方法清除马赛克 |                                        |
| --tr_blur | 传统方法模糊尺寸 |            10            |
|    --tr_down    |    传统方法下采样尺寸    |         10  |
| --medfilt_num | medfilt window of mosaic movement in the video | 11 |

* 风格转换

|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --output_size | 输出媒体的尺寸，如果是０则为原始尺寸 |512|