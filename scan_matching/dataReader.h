#pragma once
#include <iostream>
#include <cstdio>
#include <scan_matching/cpu.h>
#include <scan_matching/common.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace ScanMatching {
	namespace CPU {
		float* readPointCloudPly(std::string plyfile, int* num_points);
	}
}