#pragma once
#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <glad/glad.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

class ShaderProgram {
private:
	GLuint programId;
	GLuint vertexShaderId;
	GLuint fragmentShaderId;

	static GLuint loadShader(std::string fileName, GLuint type);
public:
	ShaderProgram(std::string vertexFileName, std::string fragmentFileName);
	void start();
	void stop();
	void cleanUp();
protected:
	virtual void bindAttributes() = 0;
	void bindAttribute(GLuint attributeIndex, std::string variableName);
};

#endif // !SHADERPROGRAM_H

