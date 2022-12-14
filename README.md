# WSWTNN-PnP：Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection

<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs1/1.png" width="90%"> </p>

Matlab implementation of "Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection''. 
## *Highlights:*
* We first formulate an implicit regularizer by plugging a denoising neural network (termed as deep denoiser), which can learn deep image priors from a large number of natural images.
* Then, we use the weighted sum of weighted tensor nuclear norm for more accurate background estimation. 
* Finally, alternating direction multiplier method is used to solve the model under the plug-and-play framework. By integrating low-rank prior with deep denoiser prior, our model achieves higher accuracy. Experiments on different scenes demonstrate that our method achieves an improved performance in terms of visual effects and quantitative metrics.

## Preparation:
#### 1. Requirement:
* Matlab 2018a, Matconvnet, Visual Studio 2015, CUDA 9.0, CUDNN v7.6.5, Windows 10, Intel Core i7-10870H CPU (2.20 GHz), RTX 2060.
* For computers with different performance and different versions of matlab, the CUDA version should be different. You need to select the version of CUDA and CUDNN according to your computer.
* For more details about code reproduction, please see the document in https://liuting20a.github.io/WSWTNN-PnP/instructions.pdf

## Test on the data:
* Run `Demo_WSWTNN_PnP.m` to perform test on data.
* The experimnet result files will be saved to `./results/`.

## The code for compared methods
* Codes for SMSL and TV-PCP methods can be found at https://wang-xiaoyang.github.io/publications/.

* Code for ASTTV-NTLA method can be found at  https://github.com/LiuTing20a/ASTTV-NTLA.

* Code for PSTNN method can be found at https://github.com/Lanneeee.

* Code for NTFRA mathod can be found at https://github.com/Electromagnetism-dog-technology/Infrared-Small-Target-Detection-via-Nonconvex-Tensor-Fibered-Rank-Approximation
* Many papers, codes and datasets of existing infrared small target detection methods can be found at https://github.com/Tianfang-Zhang/awesome-infrared-small-targets.

* Note: If you used the above codes, please cite the relevant paper.

## Results:
*To demonstrate the advantages of the WSWTNN-PnP method, we compare it with other ten methods on six different real infrared image scenes.*
### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs1/2.png" width="90%"> </p>
<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs1/3.png" width="90%"> </p>

### Ablation Experiments 
<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs1/4.jpg" width="90%"> </p>

## Details
For details such as parameter setting, please refer to [<a href="https://doi.org/10.1016/j.patcog.2022.109184">pdf</a>].

## Citation

```
@article{liu2022combining,
  title={Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection},
  author={Liu, Ting and Yin, Qian and Yang, Jungang and Wang, Yingqian and An, Wei},
  journal={Pattern Recognition},
  pages={109184},
  year={2022},
  publisher={Elsevier}
}
```
## Contact
**Welcome to raise issues or email to liuting@nudt.edu.cn for any question regarding this work.**
