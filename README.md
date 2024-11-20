# x16-Octree-Bassed-Super-Resolution
This repository contains the code for the x16 Octree-Based Super-Resolution algorithm

# References
For inspiration, we used the following sources:
1. For the 2D-3D fusion: https://github.com/tldr-group/SuperRes
2. For the Octree-Based algorithm: https://github.com/lmb-freiburg/ogn
3. For Octree-Based implementation: https://github.com/NVIDIA/MinkowskiEngine

# Introduction
This work is based on our previous research https://github.com/EvgenyUgolkov/8x_Super-Resolution/tree/main, where we worked with 8x Super-Resolution.  

In this work, we applied the Octree-Based Convolutional Neural Networks (CNN) for the Super-Resolution problem applied to the micro-CT images of rocks. We achieved 16x Super-Resolution. We demonstrate the capabilities on the example of Berea sandstone.

# Algorithm description
The micro-CT rock structure representation is highly favorable for Octree structure application. In this work, we utilize the Octree representation to save computational resources and decrease GPU memory requirements. We train the algorithm with Stages, with Progressive Growing technique. On each Stage, we memorize "dense" regions, remove them from the computations, and process "mixed" regions only for the next Stage. Schematically, the algorithm looks as follow  

![Schematic algorithm description](GH_image/alg.jpg)

For more details, please, reference our paper

# Results demonstration  
Images with arbitrary large size can be generated from the chunks of LR input. Here, we Super-Resolved LR cube with side 256 and resolution 7 um/voxel into SR cube with side 4096 and resolution 0.4375 um/voxel. The presented algorithm increases resolution, inserts sub-micron porosity, and corrects segmentation

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

