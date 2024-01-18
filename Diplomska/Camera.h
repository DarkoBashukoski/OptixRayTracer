#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>
#include <corecrt_math.h>

#include <GLFW/glfw3.h>
#include "UtilityMathFunctions.h"
#include "Timer.h"

#include <iostream>
using namespace std;

class Camera {
private:
	GLFWwindow* window;
	float3 position;
	float3 rotation;
	float moveSpeed;
	float3 up;
	float3 forward;
	float fov;
	float nearClip;
	float farClip;
	mat4 projectionMatirx;
	mat4 viewMatrix;
	double2 lastMousePos;
	int lastTab;
	bool cursorEnabled;
	double mouseSensitivity;
	void calculateProjectionMatrix();
	void calculateViewMatrix();
public:
	Camera(float3& _position, float3& _rotation, GLFWwindow* _window);
	float3& getPosition();
	float3& getRotation();
	float3& getUp();
	float getFov();
	void UVWFrame(float3& U, float3& V, float3& W);
	bool update();
	mat4& getProjectionMatrix();
	mat4& getViewMatrix();
};

#endif // !CAMERA_H


