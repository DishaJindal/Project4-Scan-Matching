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
		void buildHost(glm::vec4 *tree, glm::vec3 *points, int ynum, int size) {
			glm::vec4 *treeH = new glm::vec4[ynum];
			glm::vec3 *pointsH = new glm::vec3[ynum]; 
			cudaMemcpy(pointsH, points, ynum * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			buildTreeH(treeH, pointsH, 0, 0, 0, ynum - 1);
			cudaMemcpy(tree, treeH, ynum * sizeof(glm::vec4), cudaMemcpyHostToDevice);
		}
		__device__ void push(context* context_stack, int *top, context cur_context) {
			//printf("PUSH: %d, %d, %d\n", cur_context.idx, cur_context.dim, cur_context.good);
			context_stack[(*top)++] = cur_context;
			//printf("After PUSH: %d, %d, %d\n", context_stack[*idx - 1].idx, context_stack[*idx - 1].dim, context_stack[*idx - 1].good);
		}

		__device__ context pop(context* context_stack, int *top) {
			//printf("Before POP: %d, %d, %d\n", context_stack[*idx].idx, context_stack[*idx].dim, context_stack[*idx].good);
			return context_stack[(*top)--];
		}

		__device__ long potential_best_dist(glm::vec3 root_node, int dim, glm::vec3 query) {
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
			return dist;
		}

		__device__ void find_1NN(glm::vec4* tree, const glm::vec3 query, long nn_dist, glm::vec3* nn, const int size, context* context_stack) {
			int top = 0;
			context cur_context;
			cur_context.dim = 1;
			cur_context.good = true;
			cur_context.idx = 0;
			push(context_stack, &top, cur_context);
			while (top > 0) {
				context popped_context = pop(context_stack, &top);
				// Null Check
				if (tree[popped_context.idx].w >= 0.5) {
					glm::vec3 root_node = glm::vec3(tree[popped_context.idx].x, tree[popped_context.idx].y, tree[popped_context.idx].z);
					float dist = glm::distance(root_node, query);
					if (nn_dist > dist) {
						nn_dist = dist;
						(*nn).x = root_node.x;
						(*nn).y = root_node.y;
						(*nn).z = root_node.z;
					}
					// Prune
					if (!popped_context.good && potential_best_dist(root_node, popped_context.dim, query) > nn_dist) {
						continue;
					}
					// If Good Path or Bad Path with potential goodness

					// Good and Bad Side Calculation: Slightly Verbose
					int left = 2 * popped_context.idx + 1;
					int right = 2 * popped_context.idx + 2;
					int good_idx = -1;
					int bad_idx = -1;
					if (popped_context.dim == 0) {
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
					if (popped_context.dim == 1) {
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
					if (popped_context.dim == 2) {
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
					if (bad_idx != -1) {
						//context bad_context = { bad_idx, false, (popped_context.dim + 1) % 3 };
						context bad_context;
						bad_context.dim = (popped_context.dim + 1) % 3;
						bad_context.good = false;
						bad_context.idx = bad_idx;
						push(context_stack, &top, bad_context);
					}
					if (good_idx != -1) {
						context good_context;
						cur_context.dim = (popped_context.dim + 1) % 3;
						cur_context.good = true;
						cur_context.idx = good_idx;
						//context good_context = { good_idx, true, (popped_context.dim + 1) % 3 };
						push(context_stack, &top, good_context);

					}
				}
			}
		}

		// Kernel to find correspondence of one point in xp and update in cyp
		__global__ void kernel_find_correspondences(glm::vec4* tree, float* xp, float* cyp, int xnum, int size, int height, context* stack) {
			int i = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (i < xnum) {
				glm::vec3 nn;
				find_1NN(tree, glm::vec3(xp[3 * i], xp[3 * i + 1], xp[3 * i + 2]), LONG_MAX, &nn, size, stack + i* height);
				cyp[3 * i] = nn.x;
				cyp[3 * i + 1] = nn.y;
				cyp[3 * i + 2] = nn.z;
			}
		}

		// Finds correspondences for all points in xp from the KD Tree: tree
		void find_correspondences(float* xp, glm::vec4* tree, float* cyp, int xnum, int ynum, int blockSize, context* stack){
			dim3 xnumBlocks((xnum + blockSize - 1) / blockSize);
			int size = 1 << ilog2ceil(ynum);
			kernel_find_correspondences << <xnumBlocks, blockSize >> > (tree, xp, cyp, xnum, size, ilog2ceil(ynum), stack);
		}
	}
}
