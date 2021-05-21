# SWAG

![pR2pRs](/Figs/pR2pRs.jpg)

This repository contains the source code accompanying our CVPR 2021 paper.

**[Rethinking and Improving the Robustness of Image Style Transfer](https://arxiv.org/pdf/2104.05623.pdf)**  
[Pei Wang](http://www.svcl.ucsd.edu/~peiwang), [Yijun Li](https://yijunmaverick.github.io/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno).  
In CVPR, 2021.

```
@InProceedings{wang2021rethinking,
author = {Wang, Pei and Li, Yijun and Vasconcelos, Nuno},
title = {Rethinking and Improving the Robustness of Image Style Transfer},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 1.1. The other versions should work.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Running
To produce the results on resnet and resnet with swag
```
python resnet_swag.py --arch resnet50
python resnet_swag.py --arch resnet50_swag
```

## Disclaimer

For questions, feel free to reach out
```
Pei Wang: peiwang062@gmail.com
```

