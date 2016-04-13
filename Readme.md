# vdetlib - Python library for object detection in videos

## Introduction

The `vdetlib` python library serves to detection objects in videos. It was originally developed for the [ImageNet VID](http://image-net.org/challenges/LSVRC/2015/index#vid) challenge introduced in [ILSVRC2015](http://image-net.org/challenges/LSVRC/2015/). It contains components such as **region proposal**, **still-image object detection**, **generic object tracking**, **spatial max-pooling** and **temporal convolution**.


The [T-CNN](https://github.com/myfavouritekk/T-CNN) framework contains many tools that utilizes `vdetlib`. Please checkout that repository if you are interested.

## Citing vdetlib
If you find vdetlib useful in your research and related project, please consider citing the following work accepted in CVPR 2016.

```latex
@inproceedings{kang2016object,
  Title = {Object Detection from Video Tubelets with Convolutional Neural Networks},
  Author = {Kang, Kai and Ouyang, Wanli and Li, Hongsheng and Wang, Xiaogang},
  Booktitle = {CVPR},
  Year = {2016}
}
```

## License
This project is released under the MIT License.

## Installations
### Prerequisites
1. [caffe](https://github.com/BVLC/caffe) with `Python layer` and `pycaffe`
2. [FCN tracker](https://github.com/scott89/FCNT)
3. `Matlab` with python [engine](http://www.mathworks.com/help/matlab/matlab-engine-for-python.html?refresh=true)

### Instructions
1. Clone the repository

    ```bash
        $ git clone https://github.com/myfavouritekk/vdetlib.git
    ```

2. Compilation

    ```bash
        $ cd vdetlib
        $ make
    ```
    
## Protocols
There are some basic protocol types for using this library. All of them are defined as python dictionaries and are saved as `JSON` files. The definitions are written in the [`protocol.py`](utils/protocol.py).


## To-do list
- [ ] detailed documentation
- [ ] demo script
