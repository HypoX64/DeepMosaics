## Introduction to options
If you need more effects,  use '--option your-parameters' to enter what you need.

### Base
|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
|  --use_gpu   |   if -1, do not use gpu    |                    0                    |
| --media_path | your videos or images path |            ./imgs/ruoruo.jpg            |
|    --mode    |    program running mode(auto/clean/add/style)    |                 'auto'                  |
| --model_path |   pretrained model path    | ./pretrained_models/mosaic/add_face.pth |
| --result_dir |  output media will be saved here|                 ./result          |
|    --fps    |    read and output fps, if 0-> origin    |                 0                  |

### AddMosaic
|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --mosaic_mod | type of mosaic -> squa_avg/ squa_random/ squa_avg_circle_edge/ rect_avg/random |                    squa_avg                    |
| --mosaic_size | mosaic size,if 0 -> auto size |            0            |
|    --mask_extend    |    extend mosaic area    |         10  |
| --mask_threshold | threshold of recognize mosaic position 0~255 | 64 |

### CleanMosaic
|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --traditional | if specified, use traditional image processing methods to clean mosaic |                                        |
| --tr_blur | ksize of blur when using traditional method, it will affect final quality |            10            |
|    --tr_down    |    downsample when using traditional method,it will affect final quality    |         10  |
| --medfilt_num | medfilt window of mosaic movement in the video | 11 |

### Style Transfer
|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --output_size | size of output media, if 0 -> origin |512|