#pragma once
namespace ScanMatching {
	namespace GPU {

		void init(int xnum);

		void icp(float* xp, float* yp, int xnum, int ynum);
	}
}