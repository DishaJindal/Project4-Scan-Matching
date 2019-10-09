CUDA Scan Matching
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)

## Overview
This project implements **Iterative Closest Point Algorithm** for scan matching. 

### Contents
* `data/` Point cloud data of different objects in .ply format
* `external/` Includes and static libraries for 3rd party libraries
* `scan_matching/` Scan Matching related C++/CUDA source files 
  - `cpu.cu`:
  - `gpu.cu`:
  - `kdtree.cu`:
  - `svd3.h`: 
  - `common.cu`:
* `src/` Helper and Visualization related C++/CUDA source files
  - `main.cpp` : Setup 
  - `kernel.cu`:
  - `utilityCore.cpp`:
  - `glslUtility.cpp`:
  
## Features
   - [x] CPU Implementation of scan matching
   - [x] Naive GPU Implementation of scan matching
   - [x] KD-Tree GPU Implementation of scan matching
   - [x] Performance Analysis

## Implementations

**Flags**:
```
  #define NAIVE 0
  #define NAIVE_GPU 1
  #define KDTREE 1
  
```
### CPU 

### Naive GPU 

### KD-Tree 
