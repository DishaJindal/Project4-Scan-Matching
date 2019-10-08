#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "glm/glm.hpp"
#include "kdtree.h"

namespace ScanMatching {
	namespace GPU {

		__device__ void print_k(glm::vec3* points, int num) {
			for (int i = 0; i < num; i++) {
				printf("%f\t%f\t%f\n", points[i].x, points[i].y, points[i].z);
			}
		}

		__global__ void print_kernel(float* points, int num) {
			for (int i = 0; i < num; i++) {
				printf("%f\t%f\t%f\n", points[3 * i], points[3 * i + 1], points[3 * i + 2]);
			}
		}

		__global__ void print_v4_kernel(glm::vec4* points, int num) {
			for (int i = 0; i < num; i++) {
				printf("%f\t%f\t%f\n", points[i].x, points[i].y, points[i].z);
			}
		}

		__global__ void print_v3_kernel(glm::vec3* points, int num) {
			for (int i = 0; i < num; i++) {
				printf("%f\t%f\t%f\n", points[i].x, points[i].y, points[i].z);
			}
		}


		struct XComparator {
			__host__ __device__ inline bool operator() (const glm::vec3 a, const glm::vec3 b) {
				return a.x < b.x;
			}
		};
		struct YComparator {
			__host__ __device__ inline bool operator() (const glm::vec3 a, const glm::vec3 b) {
				return a.y < b.y;
			}
		};
		struct ZComparator {
			__host__ __device__ inline bool operator() (const glm::vec3 a, const glm::vec3 b) {
				return a.z < b.z;
			}
		};

		__device__ void buildTree(glm::vec4 *tree, glm::vec3 *points, int dim, int idx, int s, int e) {
			if (s > e)
				return;
			if (dim == 0)
				thrust::sort(thrust::device, points + s, points + e, XComparator());
			if (dim == 1)
				thrust::sort(thrust::device, points + s, points + e, YComparator());
			if (dim == 2)
				thrust::sort(thrust::device, points + s, points + e, ZComparator());
			int mid = (s + e) / 2;
			tree[idx] = glm::vec4(points[mid].x, points[mid].y, points[mid].z, 1.0f);
			buildTree(tree, points, (dim + 1) % 3, 2 * idx + 1, s, mid - 1);
			buildTree(tree, points, (dim + 1) % 3, 2 * idx + 2, mid + 1, e);
		}

		__global__ void kernel_build_tree(glm::vec4* tree, glm::vec3* ypoints_vec, int xnum) {
			buildTree(tree, ypoints_vec, 0, 0, 0, xnum - 1);
		}

		void build(glm::vec4 *tree, glm::vec3 *points, int xnum) {
			kernel_build_tree << <1, 1 >> > (tree, points, xnum);
		}
	}
}
