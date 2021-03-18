# CSE 455 final project - ResDepth
Seth Vanderwilt
3/17/21

For my final project I've been working with  **ResDepth: Learned Residual Stereo Reconstruction**, a new deep learning method to refine stereo reconstruction from Corinne Stucker and Konrad Schindler at ETH Zürich. The goal of this project is to train a network to refine DEMs that we generate from high-resolution satellite images, i.e. given two stereo pairs produce a DEM and then refine it based on some learned priors.

ResDepth description from the authors:
> We propose an embarrassingly simple but very effective scheme for high-quality dense stereo reconstruction: (i) generate an approximate reconstruction with your favourite stereo matcher; (ii) rewarp the input images with that approximate model; (iii) with the initial reconstruction and the warped images as input, train a deep network to enhance the reconstruction by regressing a residual correction; and (iv) if desired, iterate the refinement with the new, improved reconstruction.

## Problem Setup

### Satellite stereo image pairs + conventional stereo DEMs (training data, would be the inputs in production)
We have very high resolution imagery from DigitalGlobe WorldView satellites (up to even ~0.3m per pixel). If we perform stereo matching with overlapping pairs of these images, we can create 3D reconstructions & digital elevation models (DEMs). 

### LiDAR data (ground truth)
LiDAR data: in August-September 2015, Quantum Spatial flew around Mount Baker with an airborne LiDAR sensor under contract with the United States Geological Survey (USGS). Using their point cloud files, we can generate even higher resolution DEMs. Stucker & Schindler use a 2.5D CAD model of the city of Zürich for their ground truth data (with a stated height accuracy of ± 0.2 m on buildings and ±0.4 m for "general terrain").

### What the network does (output refined DEMs)
Given the stereo pair (orthorectified grayscale image rasters) and conventionally-generated DEM (also a 1-channel raster) as inputs, the network should produce a residual depth map that can then be applied to the input DEM to produce a smoother, more accurate result. We may provide further input data to the network, e.g. color/multispectral channels for the stereo images and/or land cover classifications (each pixel assigned to tree, snow, road, etc.).

## Dataset
TODO- working on the Easton glacier tile 10UEU8598

## Code
Copies in this repo will be outdated (using Colab directly)

### What I've implemented
* I implemented the network in Figure 2 of the paper in PyTorch. To start I am assuming 128x128 inputs and thus using fewer layers, as described in the paper.
TODO ![Figure2]()
* Work in progress: dataset loading & matching tools to get the stereo pairs + DEM + ground truth DEM

### I've used the following existing Python libraries so far...
* holoviz
* rasterio, rioxarray
* [demcoreg](https://github.com/dshean/demcoreg) (co-register DEMs/point clouds, tool from my advisor, David Shean!)
* pytorch/torchvision
* pytorch-lightning (for cleaner code structure, training loop, logging etc.)
* TODO experiment tracking

## Experiments

### Initial smoke test with random generated data (garbage)
To test that the network runs and is capable of memorizing something, I've passed in `torch.rand` inputs just to check that the model can do forward and backward passes without breaking.

### Small sample input test
TODO

### Full Easton glacier tile test
TODO

## Discussion
This is a work-in-progress, as I have not yet gotten our LiDAR & satellite data into a nice training & validation dataset for experiments.
Currently working on converting these to tiles with identical resolution for the first round of experiments...
I look forward to seeing the official ResDepth publication and code release at some point, as even with correct interpretation and the same training data I'm sure our models would have some differences in hyperparameter settings etc.

## References
ResDepth: Learned Residual Stereo Reconstruction. Corinne Stucker, Konrad Schindler; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, pp. 184-185. https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Stucker_ResDepth_Learned_Residual_Stereo_Reconstruction_CVPRW_2020_paper.pdf
