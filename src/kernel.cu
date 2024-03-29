#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include "device_launch_parameters.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

float *dev_pos;
float *dev_color;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

__global__ void kernAddColor(float* dev_color, int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < N) {
		dev_color[3 * i] = 0.9f;;
		dev_color[3 * i + 1] = 1.0f;
		dev_color[3 * i + 2] = 0.3f;
	}
}

__global__ void kernAddColor2(float* dev_color, int N1, int N2) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < N2) {
		dev_color[3 * N1 + 3 * i] = 1.0f;;
		dev_color[3 * N1 + 3 * i + 1] = 0.2f;
		dev_color[3 * N1 + 3 * i + 2] = 0.0f;
	}
}

void ScanMatching::copyToDevice(int N1, int N2, float* xpoints, float* ypoints) {
	cudaMemcpy(dev_pos, xpoints, sizeof(float) * 3 * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos + (3 * N1), ypoints, sizeof(float) * 3 * N2, cudaMemcpyHostToDevice);
}
/**
* Initialize memory, update some globals
*/
void ScanMatching::initSimulation(int N1, int N2, float* xpoints, float* ypoints) {
  int N = N1 + N2;
  numObjects = N;

  cudaMalloc((void**)&dev_pos, 3 * N * sizeof(float));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_color, 3 * N * sizeof(float));
  checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

  ScanMatching::copyToDevice(N1, N2, xpoints, ypoints);

  dim3 fullBlocksPerGrid1((N1 + blockSize - 1) / blockSize);
  kernAddColor << <fullBlocksPerGrid1, blockSize >> > (dev_color, N1);
  dim3 fullBlocksPerGrid2((N2 + blockSize - 1) / blockSize);
  kernAddColor2 << <fullBlocksPerGrid2, blockSize >> > (dev_color, N1, N2);
  cudaDeviceSynchronize();
}

float* ScanMatching::getDevPos() {
	return dev_pos;
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, float *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[3 * index] * c_scale;
    vbo[4 * index + 1] = pos[3 * index + 1] * c_scale;
    vbo[4 * index + 2] = pos[3 * index + 2] * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, float *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[3 * index] + 0.3f;
    vbo[4 * index + 1] = vel[3 * index + 1] + 0.3f;
    vbo[4 * index + 2] = vel[3 * index + 2] + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void ScanMatching::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_color, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}

void ScanMatching::endSimulation() {
	cudaFree(dev_pos);
	cudaFree(dev_color);
}