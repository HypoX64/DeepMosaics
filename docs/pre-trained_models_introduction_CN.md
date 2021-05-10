## 预训练模型说明
当前的预训练模型分为两类——添加/移除马赛克以及风格转换.
可以通过以下方式下载预训练模型 [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs)  [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ) <br>

### 马赛克

|              文件名              |                     描述                      |
| :------------------------------: | :-------------------------------------------: |
|           add_face.pth           |           对图片或视频中的脸部打码            |
|        clean_face_HD.pth         | 对图片或视频中的脸部去码<br>(要求内存 > 8GB). |
|         add_youknow.pth          |          对图片或视频中的...内容打码          |
| clean_youknow_resnet_9blocks.pth |          对图片或视频中的...内容去码          |
|     clean_youknow_video.pth      |             对视频中的...内容去码,推荐使用带有'video'的模型去除视频中的马赛克               |


### 风格转换

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

