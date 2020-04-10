## 参数说明
如果需要更多的效果,  请按照 '--option your-parameters' 输入所需要的参数

### 基本
|    选项    |        描述         |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
|  --use_gpu   |   if -1, do not use gpu    |                    0                    |
| --media_path | 需要处理的视频或者照片的路径 |            ./imgs/ruoruo.jpg            |
|    --mode    |    运行模式(auto/clean/add/style)    |                 'auto'                  |
| --model_path |   预训练模型的路径    | ./pretrained_models/mosaic/add_face.pth |
| --result_dir | 保存路径 |                 ./result          |
|    --fps    |    限制视频输出的fps，0则为默认    |                 0                  |

### 添加马赛克
|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --mosaic_mod | 马赛克类型 -> squa_avg/ squa_random/ squa_avg_circle_edge/ rect_avg/random |                    squa_avg                    |
| --mosaic_size | 马赛克大小，０则为自动 |            0            |
|    --mask_extend    |    拓展马赛克区域    |         10  |
| --mask_threshold | 马赛克区域识别阈值 0~255 | 64 |

### 去除马赛克
|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --traditional | 如果输入这个参数则使用传统方法清除马赛克 |                                        |
| --tr_blur | 传统方法模糊尺寸 |            10            |
|    --tr_down    |    传统方法下采样尺寸    |         10  |
| --medfilt_num | medfilt window of mosaic movement in the video | 11 |

### 风格转换
|    选项    |        描述       |                 默认                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --output_size | 输出媒体的尺寸，如果是０则为原始尺寸 |512|