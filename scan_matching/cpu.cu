#include "common.h"
#include "cpu.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "svd3.h"
#include <iostream>
#include <stdio.h>

namespace ScanMatching {
	namespace CPU {
		float *cyp, *tcyp, *m, *u, *s, *v, *tv, *R, *R1, *T, *rxmean, *xr, *xm;
		void print(float* points, int num) {
			for (int i = 0; i <= 3 * num - 3; i += 3) {
				std::cout << points[i] << "\t" << points[i + 1] << "\t" << points[i + 2] << "\n";
			}
		}

		void findCorrespondences(float* xp, float* yp, float* cyp, int xnum, int ynum) {
			for (int i = 0; i <= 3 * xnum - 3; i += 3) {
				float min_distance = FLT_MAX;
				for (int j = 0; j <= 3 * ynum - 3; j += 3) {
					float dist = glm::distance(glm::vec3(xp[i], xp[i + 1], xp[i + 2]), glm::vec3(yp[j], yp[j + 1], yp[j + 2]));
					if (dist < min_distance) {
						cyp[i] = yp[j];
						cyp[i + 1] = yp[j + 1];
						cyp[i + 2] = yp[j + 2];
						min_distance = dist;
					}
				}
			}
		}

		float* meanCenter(float* points_mean, float* points, int num_points) {
			float* mean = new float[3];
			mean[0] = 0.0f;
			mean[1] = 0.0f;
			mean[2] = 0.0f;
			for (int i = 0; i < num_points; i++) {
				mean[0] += points[3 * i];
				mean[1] += points[3 * i + 1];
				mean[2] += points[3 * i + 2];
			}
			mean[0] /= num_points;
			mean[1] /= num_points;
			mean[2] /= num_points;
			for (int i = 0; i < num_points; i++) {
				points_mean[3 * i] = points[3 * i] - mean[0];
				points_mean[3 * i + 1] = points[3 * i + 1] - mean[1];
				points_mean[3 * i + 2] = points[3 * i + 2] - mean[2];
			}
			return mean;
		}

		void transpose(float* mat, int rows, int cols, float* trans_mat) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					trans_mat[rows*j + i] = mat[cols*i + j];
				}
			}
		}

		void matrix_multiplication(const float *mat1, const float *mat2, float *mat, int m, int n, int p) {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < p; j++) {
					mat[p * i + j] = 0;
					for (int k = 0; k < n; k++) {
						mat[p * i + j] += mat1[n * i + k] * mat2[p * k + j];
					}
				}
			}
		}

		void subtract(const float *v1, const float *v2, int length, float *v) {
			for (int i = 0; i < length; i++) {
				v[i] = v1[i] - v2[i];
			}
		}

		void add_translation(float *xp, const float *xr, const float* T, int xnum) {
			for (int i = 0; i < xnum; i++) {
				xp[3 * i] = xr[3 * i] + T[0];
				xp[3 * i + 1] = xr[3 * i + 1] + T[1];
				xp[3 * i + 2] = xr[3 * i + 2] + T[2];
			}
		}

		void initialize_zero(float* arr, int n) {
			for (int i = 0; i < n; i++) {
				arr[i] = 0.0f;
			}
		}

		void init(int xnum) {
			cyp = (float*)malloc(3 * xnum * sizeof(float));
			tcyp = (float*)malloc(3 * xnum * sizeof(float));
			m = (float*)malloc(3 * 3 * sizeof(float));
			u = (float*)malloc(3 * 3 * sizeof(float));
			s = (float*)malloc(3 * 3 * sizeof(float));
			v = (float*)malloc(3 * 3 * sizeof(float));
			tv = (float*)malloc(3 * 3 * sizeof(float));
			R = (float*)malloc(3 * 3 * sizeof(float));
			R1 = (float*)malloc(3 * 3 * sizeof(float));
			T = (float*)malloc(3 * sizeof(float));
			rxmean = (float*)malloc(3 * sizeof(float));
			xr = (float*)malloc(3 * xnum * sizeof(float));
			xm = (float*)malloc(3 * xnum * sizeof(float));
		}

		void clean() {
			free(cyp);
			free(tcyp);
			free(m);
			free(u);
			free(s);
			free(v);
			free(R);
			free(R1);
			free(T);
			free(rxmean);
			free(xr);
		}

		void matrix_multiplication_test() {
			float m1[8] = { 6.0f,	7.0f,	1.0f, -1.0f, 2.0f,	0.0f,	5.0f,	3.0f };
			float m2[12] = { 4.0f,	6.0f, -3.0f, -2.0f,	0.0f,	1.0f, -4.0f,	2.0f,	5.0f, 7.0f,	8.0f,	9.0f, };
			float *m = new float[6];
			matrix_multiplication(m1, m2, m, 2, 4, 3);
			print(m, 2);
		}

		void icp(float* xp, float* yp, int xnum, int ynum) {
			auto start = std::chrono::high_resolution_clock::now();
			//std::cout << "Finding Correspondences..\n";
			findCorrespondences(xp, yp, cyp, xnum, ynum);

			//std::cout << "Mean Centering\n";
			float* xmean = meanCenter(xm, xp, xnum);
			float* ymean = meanCenter(cyp, cyp, xnum);

			//std::cout << "SVD\n";
			transpose(cyp, xnum, 3, tcyp);
			matrix_multiplication(tcyp, xp, m, 3, xnum, 3);
			initialize_zero(u, 9);
			initialize_zero(s, 9);
			initialize_zero(v, 9);
			svd(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
				u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
				s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
				v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);

			//std::cout << "Calculating R\n";
			transpose(v, 3, 3, tv);
			matrix_multiplication(u, tv, R, 3, 3, 3);

			//std::cout << "Calculating T\n";
			matrix_multiplication(R, xmean, rxmean, 3, 3, 1);
			subtract(ymean, rxmean, 3, T);

			//std::cout << "Updating Positions\n";
			transpose(R, 3, 3, R1);
			matrix_multiplication(xp, R1, xr, xnum, 3, 3); /////// Change R to R Transpose
			add_translation(xp, xr, T, xnum);
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			std::cout << elapsed.count() * 1000 << "\n";
		}
	}
}