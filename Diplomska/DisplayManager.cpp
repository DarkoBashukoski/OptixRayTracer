#include "DisplayManager.h"

DisplayManager::DisplayManager() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, NULL, NULL);
	glfwMakeContextCurrent(window);
	gladLoadGL();
	glViewport(0, 0, WIDTH, HEIGHT);

	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void DisplayManager::updateDisplay() {
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void DisplayManager::closeDisplay() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}

bool DisplayManager::isCloseRequested() {
	return glfwWindowShouldClose(window);
}

int DisplayManager::getWidth() {
	return WIDTH;
}

int DisplayManager::getHeight() {
	return HEIGHT;
}

GLFWwindow* DisplayManager::getWindow() {
	return window;
}
