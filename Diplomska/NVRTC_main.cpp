#include <nvrtc.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main123456789() {
	string location;
	string cudaSourceCode;
	const char* fileName = "DrawSolidColor.cu";

	ifstream file = ifstream(fileName, ios::binary);
	vector<unsigned char> buffer = vector<unsigned char>(istreambuf_iterator<char>(file), {});
	cudaSourceCode.assign(buffer.begin(), buffer.end());

	cout << cudaSourceCode << endl;
	
	nvrtcProgram prog = 0;

	nvrtcCreateProgram(&prog, cudaSourceCode.c_str(), fileName, 0, NULL, NULL);

	vector<const char*> options;
	
	//const char* projectDir = ""; probably not needed
	const char* projectIncludes = "C:/Users/blaZe/source/repos/Diplomska/Diplomska/Libraries/include";
	const char* projectIncludes2 = "C:/Users/blaZe/source/repos/Diplomska/Diplomska/Libraries/include/optix";
	const char* cudaIncludes = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include";

	string projString = string("-I") + projectIncludes;
	options.push_back(projString.c_str());

	string projString2 = string("-I") + projectIncludes2;
	options.push_back(projString2.c_str());

	string cudaString = string("-I") + cudaIncludes;
	options.push_back(cudaString.c_str());

	options.push_back("--optix-ir");

	nvrtcCompileProgram(prog, options.size(), options.data());
	
	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);
	
	char* log = (char*)malloc(sizeof(char) * logSize);

	nvrtcGetProgramLog(prog, log);
	
	size_t compiledCodeSize;
	nvrtcGetOptiXIRSize(prog, &compiledCodeSize);
	char* compiledCode = new char[compiledCodeSize];
	nvrtcGetOptiXIR(prog, compiledCode);
	nvrtcDestroyProgram(&prog);

	cout << compiledCodeSize << endl;

	return 0;
}