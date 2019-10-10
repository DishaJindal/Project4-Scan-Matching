#pragma once

namespace ScanMatching {
	namespace GPU {
		struct context {
			int idx;
			bool good;
			int dim;
		};

		void build_KDTree(glm::vec4 *tree, glm::vec3 *points, int ynum, int size);

		void find_correspondences(const float* xp, glm::vec4* tree, float* cyp, int xnum, int ynum, int blockSize, context* stack);
	}
}