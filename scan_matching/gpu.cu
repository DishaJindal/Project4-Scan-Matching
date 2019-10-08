#include "common.h"
#include "cpu.h"
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


#define blockSize 128
cublasHandle_t cublas_handle;

// Reference: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// Matrix Multiplication
// nr_rows_A, nr_cols_A, nr_cols_B
void gpu_blas_mmul(const float *A, const float *B, float *C, const int nr_rows_A, const int nr_cols_A, const int nr_cols_B) {
	int lda = nr_rows_A, ldb = nr_cols_A, ldc = nr_rows_A;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, nr_rows_A, nr_cols_B, nr_cols_A, alpha, A, lda, B, ldb, beta, C, ldc);
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

		float *cyp, *tcyp, *m, *u, *s, *v, *tv, *R, *R1, *T, *rxmean, *xr;

		__global__ void print_kernel(float* points, int num) {
			for (int i = 0; i <num; i++) {
				printf("%f\t%f\t%f\n", points[3*i], points[3*i + 1], points[3*i + 2]);
			}
		}

		__global__ void print_kernel2(float* points, int xnum) {
			for (int i = 0; i < xnum; i++) {
				printf("%f\t%f\t%f\n", points[i], points[40097 + i], points[2 * 40097 + i]);
			}
		}

		void init(int xnum) {
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
			cublasCreate(&cublas_handle);
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

		void meanCenter(float* xp, float* cyp, int xnum, float *xmean, float *ymean) {
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
			subtractMean << <fullBlocksPerGrid, blockSize >> > (xp, xmean, xnum);
			subtractMean << <fullBlocksPerGrid, blockSize >> > (cyp, ymean, xnum);
			cudaFree(txp);
			cudaFree(tcyp);
		}

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
			matrix_multiplication<<<xnumBlocks, blockSize>>>(dev_m1, dev_m2, dev_m3, 2, 4, 3);
			std::cout << "MM\n";
			print_kernel << <1, 1 >> > (dev_m3, 2);
			cudaDeviceSynchronize();
		}


		void icp(float* xp, float* yp, int xnum, int ynum) {
			matrix_multiplication_test();
			std::cout << "X0\n";
			print_kernel << <1,1 >> > (xp, 2);
			cudaDeviceSynchronize();
			std::cout << "Y0\n";
			print_kernel << <1, 1 >> > (yp, 5);
			cudaDeviceSynchronize();

			std::cout << "Finding Correspondences..\n";
			dim3 xnumBlocks((xnum + blockSize - 1) / blockSize);
			dim3 totalBlocks((3*xnum + blockSize - 1) / blockSize);
			findCorrespondences << <xnumBlocks, blockSize >> > (xp, yp, cyp, xnum, ynum);
			std::cout << "Y Corr\n"; 
			print_kernel << <1, 1 >> > (cyp, 5);
			cudaDeviceSynchronize();

			std::cout << "Mean Centering\n";
			float *xmean, *ymean;
			cudaMalloc((void**)&xmean, 3 * sizeof(float));
			cudaMalloc((void**)&ymean, 3 * sizeof(float));
			cudaMemset(xmean, 0.0f, 3);
			cudaMemset(ymean, 0.0f, 3);
			meanCenter(xp, cyp, xnum, xmean, ymean);
			cudaDeviceSynchronize();
			print_kernel << <1, 1 >> > (xmean, 1);
			cudaDeviceSynchronize();
			print_kernel << <1, 1 >> > (ymean, 1);
			cudaDeviceSynchronize();

			std::cout << "Transposing correspondences\n";
			transpose << <totalBlocks, blockSize >> > (cyp, tcyp, xnum, 3);

			std::cout << "Calculating Yt,X\n";
			matrix_multiplication << <1, 9 >> > (tcyp, xp, m, 3, xnum, 3);

			std::cout << "Input\n";
			print_kernel<<<1,1>>>(m, 3);
			cudaDeviceSynchronize();
			std::cout << "SVD\n";
			cudaMemset(u, 0.0f, 9);
			cudaMemset(s, 0.0f, 9);
			cudaMemset(v, 0.0f, 9);
			callSVD << <1, 1 >> > (m, u, s, v);
			cudaDeviceSynchronize();
			std::cout << "U\n";
			print_kernel << <1, 1 >> > (u, 3);
			cudaDeviceSynchronize();
			std::cout << "V\n";
			print_kernel << <1, 1 >> > (v, 3);
			cudaDeviceSynchronize();

			std::cout << "Calculating R\n";
			transpose<<<1, 9>>>(v, tv, 3, 3);
			matrix_multiplication << <1, 9 >> > (u, tv, R, 3, 3, 3);
			print_kernel << <1, 1 >> > (R, 3);
			cudaDeviceSynchronize();

			std::cout << "Calculating T\n";
			matrix_multiplication << <1, 3 >> > (R, xmean, rxmean, 3, 3, 1);
			subtract << <1, 3 >> > (ymean, rxmean, 3, T);
			print_kernel << <1, 1 >> > (T, 1);
			cudaDeviceSynchronize();

			std::cout << "Updating Positions\n";
			float *tR;
			cudaMalloc((void**)&tR, 9 * sizeof(float));
			transpose << <1, 9 >> > (R, tR, 3, 3);
			matrix_multiplication << <totalBlocks, blockSize >> > (xp, tR, xr, xnum, 3, 3);
			add_translation<<< xnumBlocks, blockSize>>>(xp, xr, T, xnum);
			print_kernel << <1, 1 >> > (xp, 2);
			cudaDeviceSynchronize();
			std::cout << "End of this iteration\n";
			cudaDeviceSynchronize();
		}
	}
}