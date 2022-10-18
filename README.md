# WSWTNN-PnP：Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection

<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs/1.jpg" width="90%"> </p>

Matlab implementation of "Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection''. 
## *Highlights:*
#### 1. * we first formulate an implicit regularizer by plugging a denoising neural network (termed as deep denoiser), which can learn deep image priors from a large number of natural images.* 
#### 2. * Then, we use the weighted sum of weighted tensor nuclear norm for more accurate background estimation.* 
#### 3. * Finally, alternating direction multiplier method is used to solve the model under the plug-and-play framework. By integrating low-rank prior with deep denoiser prior, our model achieves higher accuracy. Experiments on different scenes demonstrate that our method achieves an improved performance in terms of visual effects and quantitative metrics.*

## Preparation:
#### 1. Requirement:
* Matlab 2018a, Matconvnet, Visual Studio 2015, CUDA 9.0, CUDNN v7.6.5, Windows 10, Intel Core i7-10870H CPU (2.20 GHz), RTX 2060.
* For computers with different performance and different versions of matlab, the CUDA version should be different. You need to select the version of CUDA and CUDNN according to your computer.
* For more details about code reproduction, please see the document in Baidu's online disk. 

## Test on the data:
* Run `Demo_WSWTNN_PnP.m` to perform test on data.
* The experimnet result files will be saved to `./results/`.

## The code for compared methods
Codes for SMSL and TV-PCP methods can be found at https://wang-xiaoyang.github.io/publications/.

Code for ASTTV-NTLA method can be found at  https://github.com/LiuTing20a/ASTTV-NTLA.

Code for PSTNN method can be found at https://github.com/Lanneeee.

Code for NTFRA mathod can be found at https://github.com/Electromagnetism-dog-technology/Infrared-Small-Target-Detection-via-Nonconvex-Tensor-Fibered-Rank-Approximation

Many papers, codes and datasets of existing infrared small target detection methods can be found at https://github.com/Tianfang-Zhang/awesome-infrared-small-targets.

Note: If you used the above codes, please cite the relevant paper.





#### 2. *To demonstrate the advantages of the WSWTNN-PnP method, we compare it with other ten methods on six different real infrared image scenes.*

<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs/2.jpg" width="90%"> </p>
<p align="center"> <img src="https://raw.github.com/LiuTing20a/WSWTNN-PnP1/master/Figs/3.jpg" width="90%"> </p>

