#include "StaticShader.h"

StaticShader::StaticShader() : ShaderProgram("vertexShader.glsl", "fragmentShader.glsl") {};

void StaticShader::bindAttributes() {
	bindAttribute(0, "position");
	bindAttribute(1, "textureCoords");
}