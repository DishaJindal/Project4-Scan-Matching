#pragma once

namespace ScanMatching {
	namespace CPU {

		void init(int xnum);

		void icp(float* xp, float* yp, int xnum, int ynum);
	}
}