CUDA Scan Matching
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)

## Scan Matching
The main objective of scan matching is to search a transformation that would align a point cloud with a reference point cloud in a consistent coordinate system. If the correspondences between two point clouds are known, that the rotation and translation matrices can be calculated in closed form. Since, we do not know the correspondences, we are using this iterative ICP algorithm to find the rotation and translation matrices which converges if the starting points are close enough. 

## Iterative Closest Point Search
This project implements Iterative Closest Point Algorithm for scan matching. Following are the main steps of the algorithm:
 - Find correspondences between two point clouds 
 - Mean center both matrices and apply SVD
 - Compute rotation R and translation T
 - Apply R and T to get the updated positions
 - Compute error and If it is greater than a threshold, repeat the above steps
 - Else stop and output final alignment

### Contents
* `data/` Point cloud data of different objects in .ply format
* `external/` Includes and static libraries for 3rd party libraries
* `scan_matching/` Scan Matching related C++/CUDA source files 
  - `cpu.cu`: CPU implementation of scan matching
  - `gpu.cu`: GPU implementation of scan matching
  - `kdtree.cu`: KD Tree build and search functions
  - `svd3.h`: SVD calculation
  - `common.cu`: Helper error and printing functions
* `src/` Helper and Visualization related C++/CUDA source files
  - `main.cpp` : Main driver file and all visualization setup 
  - `kernel.cu`: Kernel functions related to visualization, position and color updates
  - `utilityCore.cpp`: Collection/kitchen sink of generally useful functions
  - `glslUtility.cpp`: Utility namespace for loading GLSL shaders
  
## Features
   - [x] CPU Implementation of scan matching
   - [x] Naive GPU Implementation of scan matching
   - [x] KD-Tree GPU Implementation of scan matching
   - [x] Performance Analysis

## Implementations
 This section talks in detail about the three different implementations of the ICP algorithm. There are following flags in the `main.cpp` file to control the current implementation (default is set to the most optimal KDTree based GPU implementation):
  ```
    #define NAIVE 0
    #define NAIVE_GPU 0
    #define KDTREE 1
  ```
### CPU 
In the CPU implemetation, the nearest neighbour search happens sequentially one by one for each point. Also, all points in the reference point cloud are checked against each source point. 

### Naive GPU 
In this version, the nearest neighbour search for each point happens parallely by launching those many threads. This leads to a hige performance gain but still for each source point, all points in the target point cloud are checked. For implementation, first step is to find the correpondences by launching a number of threads. Then, the scan implementaion from project 2 is used to find the sum of all points which is divided by the number of points to get the mean. After mean centering the source and correspondences, we find the R and T matrices using SVD. Then, matrix multiplication is used to find the updated positions and the loop continues.

### KD-Tree 
In the above two implementations, we are looking at all the points in the target pointcloud to find the correspondence for each source point which clearly is very inefficient. The idea behind using KD-Tree is to limit this search space. A k-d tree is a space-partitioning data structure for organizing points in a k-dimensional space. All of the points in our dataset are three dimensional, so we have built a 3 dimensional tree and divided the search space something like this:

<p align="center"><img src="https://github.com/DishaJindal/Project4-Scan-Matching/blob/submission/img/3dtree.png" width="300"/> </p>

In KD Tree, we divide the two paths from each node into a good or bad path by comparing the plane with the query point's coordinate. The key idea behind KD Tree is to first traverse the best path which helps in efficient pruning of the bad paths. For implementation, the tree is represented in the form of an array where the children of a node at `i` are stored at `2 * i + 1` and `2 * i + 2`, and the parent of that node is present at `i/2`. The nearest neighbour search is implemented by using an array as a stack, corresponding push and pop fucntions are implemented and top variable is maintained to keep track of the states. 

## Performance Analysis
## Bloopers
Implace Mean Centering
