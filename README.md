# x16-Octree-Bassed-Super-Resolution
This repository contains the code for the x16 Octree-Based Super-Resolution algorithm for rocks

# References
For inspiration, we used the following sources:
1. For the 2D-3D fusion Super-Resolution algorithm: https://github.com/tldr-group/SuperRes
2. For the Octree-Based algorithm: https://github.com/lmb-freiburg/ogn
3. For Octree-Based implementation: https://github.com/NVIDIA/MinkowskiEngine

# Introduction
This work is based on our previous research https://github.com/EvgenyUgolkov/8x_Super-Resolution/tree/main, where we worked with 8x Super-Resolution    

In this work, we applied the Octree-Based Convolutional Neural Networks (CNN) for the Super-Resolution problem applied to the micro-CT images of rocks. We achieved 16x Super-Resolution. We demonstrate the capabilities on the example of Berea sandstone  

# Algorithm description
The micro-CT representation of the rock structure is highly favorable for Octree structure application. In this work, we utilize the Octree representation to save computational resources and decrease GPU memory requirements. We train the algorithm with Stages, with Progressive Growing technique. On each Stage, we memorize "dense" regions, remove them from the computations, and process "mixed" regions only for the next Stage. Schematically, the algorithm looks as follow  

![Schematic algorithm description](GH_image/alg.jpg)

For more details, please, reference our paper

# Results demonstration  
3D images with arbitrary large size can be generated from the chunks of LR input. Here, we Super-Resolved Low-Resolution cube with side 256 and resolution 7 um/voxel into Super-Resolution cube with side 4096 and resolution 0.4375 um/voxel. The presented algorithm increases resolution, inserts sub-micron porosity, and corrects segmentation

![Results](GH_image/res.jpg)

# Environment
Minkovski Engine library it tricky to install. For convenience, you may use the provided ```environment.yml``` file as follow:  
1. Create a new environment from the .yml file:
```
conda env create -f environment.yml
```
2. Activate the new environment once it’s created:
```
conda activate environment
```

# Training
The training can be launched from the ![code](code) fodler with the following command:

```
python3 Architecture.py -d test --with_rotation -g_image_path Berea_7um_3.tif -d_image_path LR_M_-1.tif LR_M_0.tif LR_M_1.tif LR_M_2.tif Berea_CSLM_clay_gen.tif
```
where  

```-d``` The name of the directory to save the Generator in, under the 'progress' directory,     

```--with_rotation``` Use this option for data augmentaion (rotations and mirrors) of the High-Resolution input,      
   
```-g_image_path``` Relative path to the Low-Resolution 3D volume inside ![data](data),    

```-d_image_path``` Relative path to the High-Resolution 2D slices for each Stage inside ![data](data);  

# Evaluation  
To use the pre-trained Generator for processing Low-Resolution image, launch the following command from the ![code](code) folder 

```
python3 Evaluation.py -d test -volume_size_to_evaluate 256 256 256 -g_image_path test_7.tif
```
where  

```-d``` The name of the directory under the 'progress' directory where the pre-trained Generator parameters were saved,    
 
```-volume_size_to_evaluate``` The size of the Low-Resolution volume to be Super-Resolved;

# Pre-trained model  
Pre-trained model for Berea sandstome can be found in ![progress/test](progress/test) folder

```-g_image_path``` Relative path to the Low-Resolution image to Super-Resolve inside ![data](data);

