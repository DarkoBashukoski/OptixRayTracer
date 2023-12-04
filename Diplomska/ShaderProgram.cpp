#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(std::string vertexFileName, std::string fragmentFileName) {
	vertexShaderId = loadShader(vertexFileName, GL_VERTEX_SHADER);
	fragmentShaderId = loadShader(fragmentFileName, GL_FRAGMENT_SHADER);
	programId = glCreateProgram();
	glAttachShader(programId, vertexShaderId);
	glAttachShader(programId, fragmentShaderId);
	glLinkProgram(programId);
	glValidateProgram(programId);
}

void ShaderProgram::start() {
	glUseProgram(programId);
}

void ShaderProgram::stop() {
	glUseProgram(0);
}

void ShaderProgram::cleanUp() {
	stop();
	glDetachShader(programId, vertexShaderId);
	glDetachShader(programId, fragmentShaderId);
	glDeleteShader(vertexShaderId);
	glDeleteShader(fragmentShaderId);
	glDeleteProgram(programId);
}

void ShaderProgram::bindAttribute(GLuint attributeIndex, std::string variableName) {
	glBindAttribLocation(programId, attributeIndex, variableName.c_str());
}

GLuint ShaderProgram::loadShader(std::string fileName, GLuint type) {
	std::ifstream file;
	file.open(fileName);
	if (file.fail()) {
		std::cout << "File " << fileName << " failed to open." << std::endl;
		throw std::invalid_argument("File failed to open.");
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	const char* shaderContent = buffer.str().c_str(); //TODO fix later
	file.close();
	GLuint shaderId = glCreateShader(type);
	glShaderSource(shaderId, 1, &shaderContent, NULL);
	glCompileShader(shaderId);
	return shaderId;
}