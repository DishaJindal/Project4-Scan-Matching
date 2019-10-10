#pragma once

namespace ScanMatching {
	namespace CPU {

		void init(int xnum);

		void clean();

		void icp(float* xp, float* yp, int xnum, int ynum);
	}
}