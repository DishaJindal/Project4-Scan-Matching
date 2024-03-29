#include "main.hpp"
#include "scan_matching/dataReader.h"
#include <iostream>
#include <cstdio>
#include <scan_matching/cpu.h>
#include <scan_matching/gpu.h>
#include <scan_matching/common.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <iomanip>
#include "glm/glm.hpp"
#include <thread>
#include <chrono>
// ================
// Configuration
// ================
#define VISUALIZE 1
#define NAIVE 0
#define NAIVE_GPU 1

int N1;
int N2;
float *xpoints;
float *ypoints;
glm::vec3 *dev_pos;

const float DT = 0.2f;

glm::vec3 translate1(1.0f, 0.1f, 0.8f);
glm::vec3 rotation1(0.0f, 0.5f, -0.2f);
glm::vec3 scale1(1.5f, 1.5f, 1.5f);
glm::mat4 transformation_mat1 = utilityCore::buildTransformationMatrix(translate1, rotation1, scale1);

glm::vec3 translate2(0.0f, 0.3f, 0.2f);
glm::vec3 rotation2(-1.0f, 0.1f, 0.3f);
glm::vec3 scale2(1.5f, 1.5f, 1.5f);
glm::mat4 transformation_mat2 = utilityCore::buildTransformationMatrix(translate2, rotation2, scale2);

void read_points_w_trans(std::string plyfile, int* num_points1 , int* num_points2) {
	std::ifstream myfile(plyfile);
	float *points;
	if (!myfile.is_open())
	{
		std::cout << "Error opening file: " << plyfile;
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
					std::string temp;
					ss >> temp;
					if (count == 2) {
						*num_points1 = std::stoi(temp);
						*num_points2 = std::stoi(temp);
						printf("Points: %d\n", *num_points1);
					}
				} while (count++ < 2);
			}
		} while (myString != "end_header");

		xpoints = (float*)malloc(3 * (*num_points1) * sizeof(float));
		ypoints = (float*)malloc(3 * (*num_points1) * sizeof(float));
		int i = 0;
		while (i < *num_points1) {
			glm::vec3 pt;
			getline(myfile, myString);
			std::istringstream ss(myString);
			ss >> pt.x >> pt.y >> pt.z;
			xpoints[3 * i] = glm::vec3(transformation_mat1 * glm::vec4(pt, 1)).x;
			xpoints[3 * i + 1] = glm::vec3(transformation_mat1 * glm::vec4(pt, 1)).y;
			xpoints[3 * i + 2] = glm::vec3(transformation_mat1 * glm::vec4(pt, 1)).z;
			ypoints[3 * i] = glm::vec3(transformation_mat2 * glm::vec4(pt, 1)).x;
			ypoints[3 * i + 1] = glm::vec3(transformation_mat2 * glm::vec4(pt, 1)).y;
			ypoints[3 * i + 2] = glm::vec3(transformation_mat2 * glm::vec4(pt, 1)).z;
			i++;
		}
	}
	std::cout << "Done Reading: " << plyfile << "\n";
}

float* readPointCloudPly(std::string plyfile, int* num_points) {
	std::ifstream myfile(plyfile);
	float *points;
	if (!myfile.is_open())
	{
		std::cout << "Error opening file: " << plyfile;
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
					std::string temp;
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
			ss >> points[3 * i] >> points[3 * i + 1] >> points[3 * i + 2];
			i++;
		}
	}
	std::cout << "Done Reading: " << plyfile << "\n";
	return points;
}
/**
* C main function.
*/
int main(int argc, char* argv[]) {
	projectName = "Scan Matching";	
	 read_points_w_trans("../data/dragon_stand/dragonStandRight_0.ply", &N1, &N2);
	// read_points_w_trans("../data/bunny/data/bun045.ply", &N1, &N2);
	if (init(N1, N2, xpoints, ypoints)) {
		mainLoop(N1, N2, xpoints, ypoints);
		ScanMatching::endSimulation();
		return 0;
	}
	else {
		return 1;
	}
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int N1, int N2, float* xpoints, float* ypoints) {
	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	std::ostringstream ss;
	ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	deviceName = ss.str();

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Initialize drawing state
	initVAO(N1 + N2);

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(boidVBO_positions);
	cudaGLRegisterBufferObject(boidVBO_velocities);

	// Initialize simulation
	ScanMatching::initSimulation(N1, N2, xpoints, ypoints);

	updateCamera();

	initShaders(program);

	glEnable(GL_DEPTH_TEST);

#if NAIVE
	ScanMatching::CPU::init(N1);
#elif NAIVE_GPU
	ScanMatching::GPU::init(N1, N2, ScanMatching::getDevPos() + 3 * N1);
#endif
	return true;
}

void initVAO(int N_FOR_VIS) {

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < N_FOR_VIS; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &boidVBO_positions);
	glGenBuffers(1, &boidVBO_velocities);
	glGenBuffers(1, &boidIBO);

	glBindVertexArray(boidVAO);

	// Bind the positions array to the boidVAO by way of the boidVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(velocitiesLocation);
	glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initShaders(GLuint * program) {
	GLint location;

	program[PROG_BOID] = glslUtility::createProgram(
		"shaders/boid.vert.glsl",
		"shaders/boid.geom.glsl",
		"shaders/boid.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_BOID]);

	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

//====================================
// Main loop
//====================================
void runCUDA(int N1, int N2, float* xpoints, float* ypoints) {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertVelocities = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

	// execute the kernel
#if NAIVE
	ScanMatching::CPU::icp(xpoints, ypoints, N1, N2);
	ScanMatching::copyToDevice(N1, N2, xpoints, ypoints);
#elif NAIVE_GPU
	ScanMatching::GPU::icp(ScanMatching::getDevPos(), ScanMatching::getDevPos() + 3 * N1, N1, N2);
#endif

#if VISUALIZE
	ScanMatching::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(boidVBO_positions);
	cudaGLUnmapBufferObject(boidVBO_velocities);
}

void mainLoop(int N1, int N2, float* xpoints, float* ypoints) {
	double fps = 0;
	double timebase = 0;
	int frame = 0;
	int iter = 0;
	while (!glfwWindowShouldClose(window)) {
		if (iter++ >= 500)
			break;
		//if(iter==2)
		//	std::this_thread::sleep_for(std::chrono::milliseconds(3000));
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA(N1, N2, xpoints, ypoints);

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(program[PROG_BOID]);
		glBindVertexArray(boidVAO);
		glPointSize((GLfloat)pointSize);
		glDrawElements(GL_POINTS, N1 + N2 + 1, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
#endif
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_BOID]);
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}
