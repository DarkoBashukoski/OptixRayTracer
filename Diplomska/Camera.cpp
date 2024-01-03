#include "Camera.h"

Camera::Camera(float3& _position, float3& _rotation, GLFWwindow* _window) : position(_position), rotation(_rotation), window(_window) {
	up = { 0.0f, 1.0f, 0.0f };
	forward = { 0.0f, 0.0f, -1.0f };
	fov = 45.0f;
	moveSpeed = 0.02f;
	nearClip = 0.1f;
	farClip = 1000.0f;
	mouseSensitivity = 0.02;
	glfwGetCursorPos(window, &lastMousePos.x, &lastMousePos.y);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	calculateProjectionMatrix();
	calculateViewMatrix();
};

float3& Camera::getPosition() {
	return position;
}

float3& Camera::getRotation() {
	return rotation;
}

float3& Camera::getUp() {
	return up;
}

float Camera::getFov() {
	return fov;
}

void Camera::calculateProjectionMatrix() {
	float aspectRatio = 16.0f / 9.0f; // TODO replace with actual width and height values
	float yScale = (float)((1.0f / tan(toRadians(fov / 2.0f))) * aspectRatio);
	float xScale = yScale / aspectRatio;
	float frustumLength = farClip - nearClip;

	projectionMatirx = mat4();
	projectionMatirx.identity();
	projectionMatirx.m11 = xScale;
	projectionMatirx.m22 = yScale;
	projectionMatirx.m33 = -((farClip + nearClip) / frustumLength);
	projectionMatirx.m43 = -1; 
	projectionMatirx.m34 = -((2 * nearClip * farClip) / frustumLength);
	projectionMatirx.m44 = 0;
}

mat4& Camera::getProjectionMatrix() {
	return projectionMatirx;
}

mat4& Camera::getViewMatrix() {
	return viewMatrix;
}

void Camera::calculateViewMatrix() {
	viewMatrix = mat4();
	viewMatrix.identity();
	viewMatrix = viewMatrix * mat4::createRotationX(toRadians(rotation.x));
	viewMatrix = viewMatrix * mat4::createRotationY(toRadians(rotation.y));
	viewMatrix = viewMatrix * mat4::createRotationZ(toRadians(rotation.z));
	float3 negativeCameraPos = -position;
	viewMatrix = viewMatrix * mat4::createTranslation(negativeCameraPos);
}

bool Camera::update() {
	//Timer::getInstance()->getDelta(); TODO include delta time, probably switch delta to be ms instead of micro
	double2 currentMousePos;
	glfwGetCursorPos(window, &currentMousePos.x, &currentMousePos.y);
	double2 delta = (currentMousePos - lastMousePos);
	lastMousePos = currentMousePos;

	bool changed = false;

	if (delta.x != 0.0 || delta.y != 0.0) {
		rotation.x += delta.y * mouseSensitivity;
		rotation.y += delta.x * mouseSensitivity;

		if (rotation.x >= 90.0f) {rotation.x = 90.0f;}
		if (rotation.x <= -90.0f) {rotation.x = -90.0f;}
		if (rotation.y >= 180.0f) { rotation.y = rotation.y - 360; }
		if (rotation.y <= -180.0f) { rotation.y = rotation.y + 360; }

		forward = normalize(make_float3(
			sin(toRadians(rotation.y)),
			0.0f,
			-cos(toRadians(rotation.y))
		));

		changed = true;
	}

	float3 right = cross(forward, up);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position += forward * moveSpeed;
		changed = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position -= forward * moveSpeed;
		changed = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position += right * moveSpeed;
		changed = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position -= right * moveSpeed;
		changed = true;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position += up * moveSpeed;
		changed = true;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		position -= up * moveSpeed;
		changed = true;
	}

	if (changed) {
		calculateProjectionMatrix();
		calculateViewMatrix();
	}
	return changed;
}
