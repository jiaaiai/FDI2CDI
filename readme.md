# FDI2CDI
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for our submitted paper __FDI2CDI: Rethinking asymmetric image deformation with post-correction for particle image velocimetry__. In this work,a post-correction method is proposed to correct the velocity results of asymmetric image deformation to second-order accuracy, aiming at reducing the random interpolation error. Tested on synthetic vector fields, synthetic particle images, and practical PIV recordings, the new PIV pipeline (asymmetric image deformation + FDI2CDI post-correction) demonstrates promising performance in terms of convergency, accuracy, robustness and practicality.

### Motivation
The FDI has less image random error caused by the image warping for the fact that only one image is warped, while the CDI exhibits lower systematic error on flow field with curved streamlines, because the resulting vector fields approximate the flow velocity in second-order accuracy.Our FDI2CDI is based on an implicit geometric relationship between asymmetric FDI and symmetric CDI.Results with reduced systematic error and limited random interpolation error are achieved, integrating the beneficial of both approaches.

## Install dependencies
```bash
conda install numpy matplotlib seaborn
conda install opencv scipy pytorch scikit-image
conda install tqdm  
```

## The experiments
* [Exp1.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp1.ipynb):The convergence experiments for our FDI2CDI
* [Exp2.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp2.ipynb):The robustness experiments for our FDI2CDI
* [Exp3-1.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp3-1.ipynb):Investigate the effect of particle displacement and particle diameters on synthetic PIV images with OF
* [Exp3-2.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp3-1.ipynb):Investigate the effect of particle displacement and particle diameters on synthetic PIV images with CC
* [Exp4-1.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp4-1.ipynb):Visualize the results on synthetic PIV images experimented with OF
* [Exp4-2.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp4-2.ipynb):Visualize the results on synthetic PIV images experimented with CC
* [Exp5-1.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp5-1.ipynb):Further verify our FDI2CDI on practical PIV recordings with Solid-body rotation
* [Exp5-2.ipynb](https://github.com/jiaaiai/FDI2CDI/blob/main/Exp5-2.ipynb):Further verify our FDI2CDI on practical PIV recordings with homography transformation

### BibTeX
```
@article{Ai2024FDI2CDI,
  title={FDI2CDI: Rethinking asymmetric image deformation with post-correction for particle image velocimetry},
  author={Ai, Jia and Chen, Zuobing and Li, Junjie and Lee, Yong},
  journal={Measurement Science and Technology (submitted)},
  year={2024},
  publisher={}
}
```

### Questions?
For any questions regarding this work, please email me at [aijia@wru.edu.com](mailto:aijia@wru.edu.com)) or corresponding author [Yong Lee](https://github.com/yongleex), email,[yongli.cv@gmail.com](yongli.cv@gmail.com).

#### Acknowledgements
Parts of the code in this repository have been adapted from the following repos:

* [OpenPIV/openpiv-python](https://github.com/OpenPIV/openpiv-python)
* [opencv/opencv](https://github.com/opencv/opencv)
* [yongleex/SBCC](https://github.com/yongleex/sbcc)
* [yongleex/DiffeomorphicPIV](https://github.com/yongleex/DiffeomorphicPIV)

