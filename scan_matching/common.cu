#include "common.h"
#include "device_launch_parameters.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

void printArray(int n, float *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%f ", a[i]);
	}
	printf("]\n");
}

void printCudaArray(int size, float* data) {
	float *d_data = new float[size];
	cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
	printArray(size, d_data, true);
}

void printCuda2DArray(int height, int width, float* data) {
	float *d_data = new float[width*height];
	cudaMemcpy(d_data, data, width*height * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
		printArray(width, d_data + i * width, true);
}
