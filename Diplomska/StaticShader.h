#pragma once
#ifndef STATICSHADER_H
#define STATICSHADER_H

#include "ShaderProgram.h"

class StaticShader : public ShaderProgram {
private:

public:
	StaticShader();
protected:
	void bindAttributes() override;
};

#endif // !STATICSHADER_H

