#pragma once


namespace ScanMatching {
	namespace GPU {

		void init(int xnum, float* ypoints);

		void icp(float* xp, float* yp, int xnum, int ynum);

	}
}