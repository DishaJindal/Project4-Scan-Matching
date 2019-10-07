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
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace ScanMatching {
	namespace CPU {
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
				while (i < *num_points) {
					getline(myfile, myString);
					std::istringstream ss(myString);
					ss >> points[3*i] >> points[3 * i + 1] >> points[3 * i + 2];
					i++;
				}
			}
			cout << "Done Reading: " << plyfile << "\n";
			return points;
		}
	}
}