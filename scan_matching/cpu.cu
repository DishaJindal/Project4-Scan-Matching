#include "common.h"
#include "cpu.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "svd.h"
#include <iostream>

namespace ScanMatching {
	namespace CPU {
		void findCorrespondences(float* xp, float* yp, float* cyp, int xnum, int ynum) {
			for (int i = 0; i <= 3 * xnum - 3; i += 3) {
				long min_distance = LONG_MAX;
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

		float* meanCenter(float* points, int num_points) {
			float* mean = new float[3];
			mean[0] = 0;
			mean[1] = 0;
			mean[2] = 0;
			for (int i = 0; i <= 3 * num_points - 3; i += 3) {
				mean[0] += points[i];
				mean[1] += points[i + 1];
				mean[2] += points[i + 2];
			}
			mean[0] /= num_points;
			mean[1] /= num_points;
			mean[2] /= num_points;
			for (int i = 0; i <= 3 * num_points - 3; i += 3) {
				points[i] -= mean[0];
				points[i + 1] -= mean[1];
				points[i + 2] -= mean[2];
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

		void icp(float* xp, float* yp, int xnum, int ynum) {
			for (int iter = 1; iter <= 1; iter++) {
				std::cout << "X0\n";
				std::cout << xp[0] << "\t" << xp[1] << "\t" << xp[2] << "\n";
				std::cout << "Y0\n";
				std::cout << yp[0] << "\t" << yp[1] << "\t" << yp[2] << "\n";
				std::cout << yp[3] << "\t" << yp[4] << "\t" << yp[5] << "\n";
				
				std::cout << "Finding Correspondences..\n";
				float * cyp = (float*)malloc(3 * xnum * sizeof(float));
				findCorrespondences(xp, yp, cyp, xnum, ynum);
				std::cout << "Y Corr\n";
				std::cout << cyp[0] << "\t" << cyp[1] << "\t" << cyp[2] << "\n";
				std::cout << cyp[3] << "\t" << cyp[4] << "\t" << cyp[5] << "\n";
				std::cout << cyp[6] << "\t" << cyp[7] << "\t" << cyp[8] << "\n";

				std::cout << "Mean Centering\n";
				float* xmean = meanCenter(xp, xnum);
				float* ymean = meanCenter(cyp, xnum);
				std::cout << xmean[0] << "\t" << xmean[1] << "\t" << xmean[2] << "\n";
				std::cout << ymean[0] << "\t" << ymean[1] << "\t" << ymean[2] << "\n";
				
				std::cout << "Transposing correspondences\n";
				float * tcyp = (float*)malloc(3 * xnum * sizeof(float));
				transpose(cyp, xnum, 3, tcyp);
				
				std::cout << "Calculating X.Yt\n";
				float * m = (float*)malloc(3 * 3 * sizeof(float));
				matrix_multiplication(tcyp, xp, m, 3, xnum, 3);
	
				std::cout << "Input\n";
				std::cout << m[0] << "\t" << m[1] << "\t" << m[2] << "\n";
				std::cout << m[3] << "\t" << m[4] << "\t" << m[5] << "\n";
				std::cout << m[6] << "\t" << m[7] << "\t" << m[8] << "\n";

				std::cout << "SVD\n";
				float *u = (float*)malloc(3 * 3 * sizeof(float));
				float *s = (float*)malloc(3 * 3 * sizeof(float));
				float *v = (float*)malloc(3 * 3 * sizeof(float));
				svd(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
					u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
					s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
					v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);

				std::cout << "Calculating R\n";
				float * tv = (float*)malloc(3 * 3 * sizeof(float));
				transpose(v, 3, 3, tv);
				float * R = (float*)malloc(3 * 3 * sizeof(float));
				matrix_multiplication(u, tv, R, 3, 3, 3);
				std::cout << "Calculating T\n";
				float * T = (float*)malloc(3 * sizeof(float));
				float * temp = (float*)malloc(3 * sizeof(float));
				matrix_multiplication(R, xmean, temp, 3, 3, 1);
				subtract(ymean, temp, 3, T);

				std::cout << "Rotation Matrix\n";
				std::cout << R[0] << "\t" << R[1] << "\t" << R[2] << "\n";
				std::cout << R[3] << "\t" << R[4] << "\t" << R[5] << "\n";
				std::cout << R[6] << "\t" << R[7] << "\t" << R[8] << "\n";

				std::cout << "Translation Matrix\n";
				std::cout << T[0] << "\t" << T[1] << "\t" << T[2] << "\n";
				//free(cyp);
				//free(tcyp);
				//free(s);
			}
		}
	}
}