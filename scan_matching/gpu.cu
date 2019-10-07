#include "common.h"
#include "cpu.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "svd.h"
#include <iostream>
#include <cublas_v2.h>


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
		void print(float* points, int num) {
			for (int i = 0; i <= 3 * num - 3; i += 3) {
				std::cout << points[i] << "\t" << points[i + 1] << "\t" << points[i + 2] << "\n";
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

		void meanCenter(float* xp, float* cyp, int xnum, float *xmean, float *ymean) {
			float *txp, *tcyp;
			cudaMalloc((void**)&txp, 3 * xnum * sizeof(float));
			cudaMalloc((void**)&tcyp, 3 * xnum * sizeof(float));
			dim3 fullBlocksPerGrid((xnum + blockSize - 1) / blockSize);

			transpose << <fullBlocksPerGrid, blockSize >> > (xp, txp, xnum, 3);
			StreamCompaction::sumArray(xnum, xmean, txp);
			StreamCompaction::sumArray(xnum, xmean + 1, txp + xnum);
			StreamCompaction::sumArray(xnum, xmean + 2, txp + 2*xnum);

			transpose << <fullBlocksPerGrid, blockSize >> > (cyp, tcyp, xnum, 3);
			StreamCompaction::sumArray(xnum, ymean, tcyp);
			StreamCompaction::sumArray(xnum, ymean + 1, tcyp + xnum);
			StreamCompaction::sumArray(xnum, ymean + 2, tcyp + 2 * xnum);
			cudaFree(txp);
			cudaFree(tcyp);

			subtractMean << <fullBlocksPerGrid, blockSize >> > (xp, xmean, xnum);
			subtractMean << <fullBlocksPerGrid, blockSize >> > (cyp, ymean, xnum);
		}

		void icp(float* xp, float* yp, int xnum, int ynum) {
			std::cout << "Finding Correspondences..\n";
			dim3 fullBlocksPerGrid((xnum + blockSize - 1) / blockSize);
			findCorrespondences << <fullBlocksPerGrid, blockSize >> > (xp, yp, cyp, xnum, ynum);

			std::cout << "Mean Centering\n";
			float *xmean, *ymean;
			cudaMalloc((void**)&xmean, 3 * sizeof(float));
			cudaMalloc((void**)&ymean, 3 * sizeof(float));
			meanCenter(xp, cyp, xnum, xmean, ymean);

			std::cout << "Transposing correspondences\n";
			transpose << <fullBlocksPerGrid, blockSize >> > (cyp, tcyp, xnum, 3);

			std::cout << "Calculating X.Yt\n";
			gpu_blas_mmul(tcyp, xp, m, 3, xnum, 3);

			std::cout << "Input\n";
			print(m, 3);
			std::cout << "SVD\n";
			cudaMemset(u, 0.0f, 9);
			cudaMemset(s, 0.0f, 9);
			cudaMemset(v, 0.0f, 9);
			/*svd(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
				u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
				s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
				v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
			std::cout << "U\n";
			print(u, 3);
			std::cout << "V\n";
			print(v, 3);

			std::cout << "Calculating R\n";
			transpose(v, 3, 3, tv);

			matrix_multiplication(u, tv, R, 3, 3, 3);
			print(R, 3);

			std::cout << "Calculating T\n";
			matrix_multiplication(R, xmean, rxmean, 3, 3, 1);
			subtract(ymean, rxmean, 3, T);
			print(T, 1);

			std::cout << "Updating Positions\n";
			matrix_multiplication(xp, R, xr, xnum, 3, 3);
			std::cout << "xr\n";
			print(xp, 1);
			print(xr, 1);
			add_translation(xp, xr, T, xnum);
			print(xp, 2);*/
		}
	}
}