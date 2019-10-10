#include "common.h"
#include "gpu.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "svd3_cuda.h"
#include <iostream>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "kdtree.h"

#define blockSize 128
#define KDTREE 1

__global__ void print_kernel(float* points, int num) {
	for (int i = 0; i < num; i++) {
		printf("%f\t%f\t%f\n", points[3 * i], points[3 * i + 1], points[3 * i + 2]);
	}
}

__global__ void print_v4_kernel(glm::vec4* points, int num) {
	for (int i = 0; i < num; i++) {
		printf("%f\t%f\t%f\t%f\n", points[i].x, points[i].y, points[i].z, points[i].w);
	}
}

__global__ void print_v3_kernel(glm::vec3* points, int num) {
	for (int i = 0; i < num; i++) {
		printf("%f\t%f\t%f\n", points[i].x, points[i].y, points[i].z);
	}
}

namespace StreamCompaction {
	__global__ void kernelUpSweepStepEfficient(int n, int d, float* cdata) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		int prev_step_size = 1 << d;
		int cur_step_size = 2 * prev_step_size;
		int new_offset = k * cur_step_size;
		cdata[new_offset + cur_step_size - 1] += cdata[new_offset + prev_step_size - 1];
	}
	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void sumArray(int n, float* sum, const float *idata) {
		// Memory Allocation and Copying
		int power_size = pow(2, ilog2ceil(n));
		float *sumArray;
		cudaMalloc((void**)&sumArray, power_size * sizeof(float));
		checkCUDAErrorFn("cudaMalloc sumArray failed!");
		cudaMemset(sumArray, 0, power_size * sizeof(float));
		cudaMemcpy(sumArray, idata, n * sizeof(float), cudaMemcpyDeviceToDevice);

		int numThreads;
		//Up Sweep
		for (int d = 0; d <= ilog2ceil(power_size) - 1; d++) {
			numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
			dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
			kernelUpSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, sumArray);
		}
		// Copy Back and Free Memory
		cudaMemcpy(sum, sumArray + power_size - 1, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaFree(sumArray);
	}
}

namespace ScanMatching {
	namespace GPU {

		float *cyp, *tcyp, *m, *u, *s, *v, *tv, *R, *R1, *T, *rxmean, *xr, *xmc;
		context *stack;
		glm::vec4 *tree;

		__global__ void print_kernel2(float* points, int xnum) {
			for (int i = 0; i < xnum; i++) {
				printf("%f\t%f\t%f\n", points[i], points[40097 + i], points[2 * 40097 + i]);
			}
		}

		__global__ void float_to_vec3(int ynum, float* points, glm::vec3* points_vec) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < ynum) {
				points_vec[i] = glm::vec3(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
			}
		}

		void init(int xnum, int ynum, float* ypoints) {
			cudaMalloc((void**)&cyp, 3 * xnum * sizeof(float));
			cudaMalloc((void**)&tcyp, 3 * xnum * sizeof(float));
			cudaMalloc((void**)&m, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&u, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&s, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&v, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&tv, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&R, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&R1, 3 * 3 * sizeof(float));
			cudaMalloc((void**)&T, 3 * sizeof(float));
			cudaMalloc((void**)&rxmean, 3 * sizeof(float));
			cudaMalloc((void**)&xr, 3 * xnum * sizeof(float));
			cudaMalloc((void**)&xmc, 3 * xnum * sizeof(float));

			// Build Tree
			#if KDTREE
				std::cout << "Building Tree\n";
				int size = 1 << ilog2ceil(ynum);
				cudaMalloc((void**)&tree, size * sizeof(glm::vec4));
				glm::vec3 *ypoints_vec;
				cudaMalloc((void**)&ypoints_vec, ynum * sizeof(glm::vec3));
				dim3 ynumBlocks((ynum + blockSize - 1) / blockSize);
				float_to_vec3 << <ynumBlocks, blockSize >> > (ynum, ypoints, ypoints_vec);
				buildHost(tree, ypoints_vec, ynum, size);
				cudaFree(ypoints_vec);
				cudaMalloc((void**)&stack, xnum *  (ilog2ceil(ynum) + 1) * sizeof(context));
				checkCUDAErrorFn("cudaMalloc stack failed!");
			#endif
		}

		__global__ void findCorrespondences(float* xp, float* yp, float* cyp, int xnum, int ynum) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < xnum) {
				float min_distance = FLT_MAX;
				for (int j = 0; j < ynum; j++) {
					float dist = glm::distance(glm::vec3(xp[3 * i], xp[3 * i + 1], xp[3 * i + 2]), glm::vec3(yp[3 * j], yp[3 * j + 1], yp[3 * j + 2]));
					if (dist < min_distance) {
						cyp[3 * i] = yp[3 * j];
						cyp[3 * i + 1] = yp[3 * j + 1];
						cyp[3 * i + 2] = yp[3 * j + 2];
						min_distance = dist;
					}
				}
			}
		}

		__global__ void transpose(float *xp, float* txp, int rows, int cols) {
			int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (idx < rows*cols) {
				int i = idx / cols;
				int j = idx % cols;
				txp[rows * j + i] = xp[cols * i + j];
			}
		}

		__global__ void subtractMean(float* xp, float* mean, int xnum) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < xnum) {
				xp[3 * i] -= mean[0];
				xp[3 * i + 1] -= mean[1];
				xp[3 * i + 2] -= mean[2];
			}
		}
		__global__ void callSVD(float* m, float* u, float* s, float* v) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i == 0) {
				svd(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
					u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
					s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
					v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
			}
		}

		__global__ void subtract(const float *v1, const float *v2, int length, float *v) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < length) {
				v[i] = v1[i] - v2[i];
			}
		}

		__global__ void add_translation(float *xp, const float *xr, const float* T, int xnum) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < xnum) {
				xp[3 * i] = xr[3 * i] + T[0];
				xp[3 * i + 1] = xr[3 * i + 1] + T[1];
				xp[3 * i + 2] = xr[3 * i + 2] + T[2];
			}
		}

		__global__ void divide(float* arr, int length, float divisor) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < length) {
				arr[i] /= divisor;
			}
		}

		__global__ void matrix_multiplication(const float *mat1, const float *mat2, float *mat, int m, int n, int p) {
			int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (idx < m*p) {
				int i = idx / p;
				int j = idx % p;
				mat[p * i + j] = 0;
				for (int k = 0; k < n; k++) {
					mat[p * i + j] += mat1[n * i + k] * mat2[p * k + j];
				}
			}
		}

		void meanCenter(float* xmc, float* xp, float* cyp, int xnum, float *xmean, float *ymean) {
			float *txp, *tcyp;
			cudaMalloc((void**)&txp, 3 * xnum * sizeof(float));
			cudaMalloc((void**)&tcyp, 3 * xnum * sizeof(float));
			dim3 fullBlocksPerGrid((3*xnum + blockSize - 1) / blockSize);
			transpose << <fullBlocksPerGrid, blockSize >> > (xp, txp, xnum, 3);
			transpose << <fullBlocksPerGrid, blockSize >> > (cyp, tcyp, xnum, 3);
			StreamCompaction::sumArray(xnum, xmean, txp);
			StreamCompaction::sumArray(xnum, xmean + 1, txp + xnum);
			StreamCompaction::sumArray(xnum, xmean + 2, txp + 2*xnum);
			StreamCompaction::sumArray(xnum, ymean, tcyp);
			StreamCompaction::sumArray(xnum, ymean + 1, tcyp + xnum);
			StreamCompaction::sumArray(xnum, ymean + 2, tcyp + 2 * xnum);
			divide << <1, 3 >> > (xmean, 3, xnum);
			divide << <1, 3 >> > (ymean, 3, xnum);
			subtractMean << <fullBlocksPerGrid, blockSize >> > (xmc, xmean, xnum);
			subtractMean << <fullBlocksPerGrid, blockSize >> > (cyp, ymean, xnum);
			cudaFree(txp);
			cudaFree(tcyp);
		}

		void icp(float* xp, float* yp, int xnum, int ynum) {
			auto start = std::chrono::high_resolution_clock::now();
			cudaMemcpy(xmc, xp, 3 * xnum * sizeof(float), cudaMemcpyDeviceToDevice);
			dim3 xnumBlocks((xnum + blockSize - 1) / blockSize);
			dim3 totalBlocks((3*xnum + blockSize - 1) / blockSize);
			#if KDTREE
				find_correspondences(xp, tree, cyp, xnum, ynum, blockSize, stack);
				checkCUDAErrorFn("find_correspondences calculation failed!");
			#else
				findCorrespondences << <xnumBlocks, blockSize >> > (xp, yp, cyp, xnum, ynum);
			#endif

			//std::cout << "Mean Centering\n";
			float *xmean, *ymean;
			cudaMalloc((void**)&xmean, 3 * sizeof(float));
			cudaMalloc((void**)&ymean, 3 * sizeof(float));
			cudaMemset(xmean, 0.0f, 3);
			cudaMemset(ymean, 0.0f, 3);
			meanCenter(xmc, xp, cyp, xnum, xmean, ymean);
			
			//std::cout << "SVD\n";
			transpose << <totalBlocks, blockSize >> > (cyp, tcyp, xnum, 3);
			matrix_multiplication << <1, 9 >> > (tcyp, xmc, m, 3, xnum, 3);
			cudaMemset(u, 0.0f, 9);
			cudaMemset(s, 0.0f, 9);
			cudaMemset(v, 0.0f, 9);
			callSVD << <1, 1 >> > (m, u, s, v);

			//std::cout << "Calculating R\n";
			transpose<<<1, 9>>>(v, tv, 3, 3);
			matrix_multiplication << <1, 9 >> > (u, tv, R, 3, 3, 3);

			//std::cout << "Calculating T\n";
			matrix_multiplication << <1, 3 >> > (R, xmean, rxmean, 3, 3, 1);
			subtract << <1, 3 >> > (ymean, rxmean, 3, T);

			//std::cout << "Updating Positions\n";
			float *tR;
			cudaMalloc((void**)&tR, 9 * sizeof(float));
			transpose << <1, 9 >> > (R, tR, 3, 3);
			matrix_multiplication << <totalBlocks, blockSize >> > (xp, tR, xr, xnum, 3, 3);
			add_translation<<< xnumBlocks, blockSize>>>(xp, xr, T, xnum);

			cudaDeviceSynchronize();
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			//std::cout << elapsed.count() * 1000 << "\n";
		}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////TEST FUNCTIONS///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
		void matrix_multiplication_test() {
			float m1[8] = { 6.0f,	7.0f,	1.0f, -1.0f, 2.0f,	0.0f,	5.0f,	3.0f };
			float m2[12] = { 4.0f,	6.0f, -3.0f, -2.0f,	0.0f,	1.0f, -4.0f,	2.0f,	5.0f, 7.0f,	8.0f,	9.0f, };
			float *dev_m1, *dev_m2, *dev_m3;
			cudaMalloc((void**)&dev_m1, 8 * sizeof(float));
			cudaMalloc((void**)&dev_m2, 12 * sizeof(float));
			cudaMalloc((void**)&dev_m3, 6 * sizeof(float));
			cudaMemcpy(dev_m1, m1, 8 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_m2, m2, 12 * sizeof(float), cudaMemcpyHostToDevice);

			dim3 xnumBlocks((12 + blockSize - 1) / blockSize);
			matrix_multiplication << <xnumBlocks, blockSize >> > (dev_m1, dev_m2, dev_m3, 2, 4, 3);
			std::cout << "MM\n";
			print_kernel << <1, 1 >> > (dev_m3, 2);
			cudaDeviceSynchronize();
		}
		__device__ int fact(int f)
		{
			// Test using 			
			// call_fact << <1, 1 >> > (5);
			if (f == 0)
				return 1;
			else
				return f * fact(f - 1);
		}
		__global__ void call_fact(int f) {
			printf("Factorial: %d\n", fact(f));
		}
	}
}