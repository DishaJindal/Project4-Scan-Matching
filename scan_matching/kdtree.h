#pragma once

namespace ScanMatching {
	namespace GPU {
		void build(glm::vec4 *tree, glm::vec3 *points, int xnum);

		void buildHost(glm::vec4 *tree, glm::vec3 *points, int xnum);

		void find_correspondences(float* xp, glm::vec4* tree, float* cyp, int xnum, int ynum, int blockSize);

	}
}