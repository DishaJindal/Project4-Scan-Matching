#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "glm/glm.hpp"
#include "kdtree.h"
#include "device_launch_parameters.h"
#include "common.h"

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
			printf("In Tree: idx: %d dim: %d, s: %d e: %d\n", idx, dim, s, e);
			print_k(points, 2);
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
			printf("Out Tree\n");
			print_k(points, 2);
		}

		__host__ void buildTreeH(glm::vec4 *tree, glm::vec3 *points, int dim, int idx, int s, int e) {
			if (s > e)
				return;
			if (dim == 0)
				thrust::sort(thrust::host, points + s, points + e, XComparator());
			if (dim == 1)
				thrust::sort(thrust::host, points + s, points + e, YComparator());
			if (dim == 2)
				thrust::sort(thrust::host, points + s, points + e, ZComparator());
			int mid = (s + e) / 2;
			tree[idx] = glm::vec4(points[mid].x, points[mid].y, points[mid].z, 1.0f);
			buildTreeH(tree, points, (dim + 1) % 3, 2 * idx + 1, s, mid - 1);
			buildTreeH(tree, points, (dim + 1) % 3, 2 * idx + 2, mid + 1, e);
		}

		__global__ void kernel_build_tree(glm::vec4* tree, glm::vec3* ypoints_vec, int ynum) {
			buildTree(tree, ypoints_vec, 0, 0, 0, ynum - 1);
		}

		// Builds KD Tree
		void build(glm::vec4 *tree, glm::vec3 *points, int xnum) {
			kernel_build_tree << <1, 1 >> > (tree, points, xnum);
		}

		// Builds KD Tree
		void buildHost(glm::vec4 *tree, glm::vec3 *points, int ynum) {
			glm::vec4 *treeH = new glm::vec4[ynum];
			glm::vec3 *pointsH = new glm::vec3[ynum]; 
			cudaMemcpy(pointsH, points, ynum * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			buildTreeH(treeH, pointsH, 0, 0, 0, ynum - 1);
			cudaMemcpy(tree, treeH, ynum * sizeof(glm::vec4), cudaMemcpyHostToDevice);
		}
		__device__ void find_1NN(glm::vec4* tree, glm::vec3 query, int root, int dim, long nn_dist, glm::vec3* nn, int size) {


		}

		__device__ void find_1NNRecursive(glm::vec4* tree, glm::vec3 query, int root, int dim, long nn_dist, glm::vec3* nn, int size) {
			// Equivalent to Null Check
			if ((tree[root].w - 0) < 1)
				return;

			// Find distance with root and Update If required
			glm::vec3 root_node = glm::vec3(tree[root].x, tree[root].y, tree[root].z);
			float dist = glm::distance(root_node, query);
			if (nn_dist > dist) {
				nn_dist = dist;
				nn = &root_node;
			}
			int left = 2 * root + 1;
			int right = 2 * root + 2;

			// Good and Bad Side Calculation: Slightly Verbose
			int good_idx = -1, bad_idx = -1;
			if (dim == 0) {
				if (left < size && tree[left].w > 0.5) {
					if (tree[left].x <= query.x)
						good_idx = left;
					else
						bad_idx = left;
				}
				if (right < size && tree[right].w > 0.5) {
					if (tree[right].x <= query.x)
						good_idx = right;
					else
						bad_idx = right;
				}
			}
			if (dim == 1) {
				if (left < size && tree[left].w > 0.5) {
					if (tree[left].y <= query.y)
						good_idx = left;
					else
						bad_idx = left;
				}
				if (right < size && tree[right].w > 0.5) {
					if (tree[right].y <= query.y)
						good_idx = right;
					else
						bad_idx = right;
				}

			}
			if (dim == 2) {
				if (left < size && tree[left].w > 0.5) {
					if (tree[left].z <= query.z)
						good_idx = left;
					else
						bad_idx = left;
				}
				if (right < size && tree[right].w > 0.5) {
					if (tree[right].z <= query.z)
						good_idx = right;
					else
						bad_idx = right;
				}
			}
			if(good_idx != -1)
				find_1NN(tree, query, good_idx, (dim + 1) % 3, nn_dist, nn, size);
			// Could we have something on the bad side
				// Distance with the best point in the bad side --> Can optimize this
				// If Big-> Prune
				// Else Recurse
			if (bad_idx != -1) {
				long dist;
				if (dim == 0) {
					dist = fabsf(query.x - root_node.x);
				}
				if (dim == 1) {
					dist = fabsf(query.y - root_node.y);
				}
				if (dim == 2) {
					dist = fabsf(query.z - root_node.z);
				}
				if (dist < nn_dist)
					find_1NN(tree, query, bad_idx, (dim + 1) % 3, nn_dist, nn, size);
			}
		}

		// Kernel to find correspondence of one point in xp and update in cyp
		__global__ void kernel_find_correspondences(glm::vec4* tree, float* xp, float* cyp, int xnum, int size) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < xnum) {
				glm::vec3 nn;
				find_1NN(tree, glm::vec3(xp[3 * i], xp[3 * i + 1], xp[3 * i + 2]), 0, 0, LONG_MAX, &nn, size);
				cyp[3 * i] = nn.x;
				cyp[3 * i + 1] = nn.y;
				cyp[3 * i + 2] = nn.z;
			}
		}

		// Finds correspondences for all points in xp from the KD Tree: tree
		void find_correspondences(float* xp, glm::vec4* tree, float* cyp, int xnum, int ynum, int blockSize){
			dim3 xnumBlocks((xnum + blockSize - 1) / blockSize);
			int size = 1 << ilog2ceil(xnum);
			kernel_find_correspondences << <xnumBlocks, blockSize >> > (tree, xp, cyp, xnum, size);
		}
	}
}
