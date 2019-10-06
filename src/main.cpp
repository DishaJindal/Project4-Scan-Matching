/**
 * @file      main.cpp
 * @brief     Scan Matching
 * @authors   Disha Jindal
 * @date      2019
 * @copyright University of Pennsylvania
 */
#include <iostream>
#include <cstdio>
#include <scan_matching/cpu.h>
#include <scan_matching/common.h>
#include "testing_helpers.hpp"
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace std;

float* readPointCloudPly(std::string plyfile, int* num_points) {
	ifstream myfile(plyfile);
	float *points;
	if (!myfile.is_open())
	{
		cout << "Error opening file: "<< plyfile;
		exit(1);
	}
	std::string myString;
	
	if (!myfile.eof())
	{
		do {
			getline(myfile, myString);
			if (!myString.compare(0, 14, "element vertex")) {
				std::istringstream ss(myString);
				int count = 0;
				do {
					string temp;
					ss >> temp;
					if (count == 2)
						*num_points = std::stoi(temp);
				} while (count++ < 2);
			}
		} while (myString != "end_header");

		points = (float*)malloc(3 * (*num_points) * sizeof(float));
		int i = 0;
		while (i < (*num_points - 3)) {
			getline(myfile, myString);
			std::istringstream ss(myString);
			ss >> points[i] >> points[i + 1] >> points[i + 2];
			i += 3;
		}
	}
	cout << "Done Reading: " << plyfile << "\n";
	return points;
}

int main(int argc, char* argv[]) {
	cout << "In Scan Matching Main\n";
	int xnum, ynum;
	float *xpoints = readPointCloudPly("C:\\Users\\djjindal\\Project4-Scan-Matching\\data\\bunny\\data\\bun000.ply", &xnum);
	float *ypoints = readPointCloudPly("C:\\Users\\djjindal\\Project4-Scan-Matching\\data\\bunny\\data\\bun045.ply", &ynum);
	ScanMatching::CPU::icp(xpoints, ypoints, xnum, ynum);
	//free(xpoints);
	//free(ypoints);
	cout << "Out of ICP\n";
	return 0;
}
