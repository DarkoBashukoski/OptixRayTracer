#pragma once
#ifndef RENDERER_H
#define RENDERER_H

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include "CudaOutputBuffer.h"
#include <vector>
#include "StaticShader.h"
#include "DisplayManager.h"

class Renderer {
private:
	std::vector<float> positions;
	std::vector<float> textureCoords;
	std::vector<GLuint> vbos;
	GLuint vaoId;
	GLuint textureId;
	StaticShader shader;

	void storeDataInAttributeList(int attributeNumber, std::vector<float>& data, int coordinateSize);
public:
	Renderer();
	~Renderer();
	void render(CudaOutputBuffer<uchar4>& buffer);
};

#endif // !RENDERER_H

