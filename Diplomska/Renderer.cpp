#include "Renderer.h"

Renderer::Renderer() {
	vbos = std::vector<GLuint>();
	shader = StaticShader();

	positions = {
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	textureCoords = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
	};

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);

	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	storeDataInAttributeList(0, positions, 3);
	storeDataInAttributeList(1, textureCoords, 2);

	glBindVertexArray(0);
}

Renderer::~Renderer() {
	glDeleteVertexArrays(1, &vaoId);
	for (GLuint vboId : vbos) {
		glDeleteBuffers(1, &vboId);
	}
}

void Renderer::storeDataInAttributeList(int attributeNumber, std::vector<float>& data, int coordinateSize) {
	GLuint vboId;
	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
	glVertexAttribPointer(attributeNumber, coordinateSize, GL_FLOAT, GL_FALSE, coordinateSize * sizeof(float), (void**)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	vbos.push_back(vboId);
}

void Renderer::render(CudaOutputBuffer<uchar4>& buffer) {
	shader.start();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1.0f, 0.0f, 1.0f, 1.0f);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	glBindVertexArray(vaoId);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	//texturing - nesto definitivno ne e vo red, black screen ponekogas, nz zosto, mojt poubo da se napisit
	
	uchar4* data = buffer.getHostPointer();

	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, DisplayManager::getInstance()->getWidth(), DisplayManager::getInstance()->getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureId);
	
	//texturing end

	glDrawArrays(GL_TRIANGLES, 0, 6);

	ImGui::Begin("My name is window, ImGUI window");
	//ImGui::ColorEdit3("Background color", );
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindVertexArray(0);

	shader.stop();
}