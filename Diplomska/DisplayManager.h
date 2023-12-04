#pragma once

#ifndef DISPLAYMANAGER_H
#define DISPLAYMANAGER_H

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

class DisplayManager {
private:
	const int WIDTH = 1920;
	const int HEIGHT = 1080;
	const char* TITLE = "Raytracing App";
	GLFWwindow* window;
public:
	DisplayManager();
	void updateDisplay();
	void closeDisplay();
	bool isCloseRequested();
	int getWidth();
	int getHeight();
	GLFWwindow* getWindow();
};

#endif // !DISPLAYMANAGER_H
