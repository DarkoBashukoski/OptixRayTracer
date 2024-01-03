#include "OptixShaderCompiler.h"
#include "ErrorChecks.h"

char* OptixShaderCompiler::compileShader(string fileName, size_t* compiledCodeSize) {
	string cudaSourceCode;

	ifstream file = ifstream(fileName, ios::binary);
	vector<unsigned char> buffer = vector<unsigned char>(istreambuf_iterator<char>(file), {});
	cudaSourceCode.assign(buffer.begin(), buffer.end());

	nvrtcProgram prog = 0;

	NVRTC_CHECK(nvrtcCreateProgram(&prog, cudaSourceCode.c_str(), fileName.c_str(), 0, NULL, NULL));

	vector<const char*> includeOptions;

	const char* projectIncludes = "C:/Users/blaZe/source/repos/Diplomska/Diplomska/Libraries/include";
	const char* projectIncludes2 = "C:/Users/blaZe/source/repos/Diplomska/Diplomska/Libraries/include/optix";
	const char* cudaIncludes = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include";

	string projString = string("-I") + projectIncludes;
	includeOptions.push_back(projString.c_str());

	string projString2 = string("-I") + projectIncludes2;
	includeOptions.push_back(projString2.c_str());

	string cudaString = string("-I") + cudaIncludes;
	includeOptions.push_back(cudaString.c_str());

	includeOptions.push_back("--optix-ir");

	NVRTC_CHECK(nvrtcCompileProgram(prog, includeOptions.size(), includeOptions.data()));

	size_t errorLogForNVRTCSize;
	NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &errorLogForNVRTCSize));
	char* nvrtcErrorlog = (char*)malloc(sizeof(char) * errorLogForNVRTCSize);
	NVRTC_CHECK(nvrtcGetProgramLog(prog, nvrtcErrorlog));
	cout << nvrtcErrorlog << endl;

	NVRTC_CHECK(nvrtcGetOptiXIRSize(prog, compiledCodeSize));
	char* compiledCode = new char[*compiledCodeSize];
	NVRTC_CHECK(nvrtcGetOptiXIR(prog, compiledCode));
	NVRTC_CHECK(nvrtcDestroyProgram(&prog));

	return compiledCode;
}