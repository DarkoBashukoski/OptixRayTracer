#pragma once
#ifndef OPTIXSHADERCOMPILER_H
#define OPTIXSHADERCOMPILER_H

#include <nvrtc.h>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

class OptixShaderCompiler {
private:

public:
	static char* compileShader(string fileName, size_t* compiledCodeSize);
};

#endif // !OPTIXSHADERCOMPILER_H
