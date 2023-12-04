/*
#include <optix/optix.h>
#include <optix/optix_function_table_definition.h>
#include <optix/optix_stack_size.h>
#include <optix/optix_stubs.h>

#include "CudaOutputBuffer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "Parameters.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <nvrtc.h>

using namespace std;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;

int main123() {
	int width = 1920;
	int height = 1080;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Test", NULL, NULL);
	glfwMakeContextCurrent(window);
	gladLoadGL();
	glViewport(0, 0, width, height);

	OptixDeviceContext context = nullptr;

	cudaFree(0);
	CUcontext cuCtx = 0;
	optixInit();
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	optixDeviceContextCreate(cuCtx, &options, &context);
	optixDeviceContextSetCacheEnabled(context, 0);

	OptixModule module = nullptr;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixModuleCompileOptions moduleCompileOptions = {};

	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	char LOG[2048];
	size_t LOG_SIZE;

	// Code compilation, should be split in separate function later

	string location;
	string cudaSourceCode;
	const char* fileName = "DrawSolidColor.cu";

	ifstream file = ifstream(fileName, ios::binary);
	vector<unsigned char> buffer = vector<unsigned char>(istreambuf_iterator<char>(file), {});
	cudaSourceCode.assign(buffer.begin(), buffer.end());

	nvrtcProgram prog = 0;

	nvrtcCreateProgram(&prog, cudaSourceCode.c_str(), fileName, 0, NULL, NULL);

	vector<const char*> includeOptions;

	//const char* projectDir = ""; probably not needed
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

	nvrtcCompileProgram(prog, includeOptions.size(), includeOptions.data());

	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);

	char* asdLog = (char*)malloc(sizeof(char) * logSize);

	nvrtcGetProgramLog(prog, asdLog);

	size_t compiledCodeSize;
	nvrtcGetOptiXIRSize(prog, &compiledCodeSize);
	char* compiledCode = new char[compiledCodeSize];
	nvrtcGetOptiXIR(prog, compiledCode);
	nvrtcDestroyProgram(&prog);

	// Code compilation end

	optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, compiledCode, compiledCodeSize, LOG, &LOG_SIZE, &module);

	OptixProgramGroup raygenProgramGroup = nullptr;
	OptixProgramGroup missProgramGroup = nullptr;

	OptixProgramGroupOptions programGroupOptions = {};

	OptixProgramGroupDesc raygenProgramGroupDescription = {};
	raygenProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenProgramGroupDescription.raygen.module = module;
	raygenProgramGroupDescription.raygen.entryFunctionName = "__raygen__drawSolidColor";
	optixProgramGroupCreate(context, &raygenProgramGroupDescription, 1, &programGroupOptions, LOG, &LOG_SIZE, &raygenProgramGroup);

	OptixProgramGroupDesc missProgramGroupDescription = {};
	missProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	optixProgramGroupCreate(context, &missProgramGroupDescription, 1, &programGroupOptions, LOG, &LOG_SIZE, &raygenProgramGroup);

	OptixPipeline pipeline = nullptr;
	OptixProgramGroup programGroups[] = { raygenProgramGroup };

	uint32_t maxTraceDepth = 0;

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
	optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, sizeof(programGroups) / sizeof(programGroups[0]), LOG, &LOG_SIZE, &pipeline);

	OptixStackSizes stackSizes = {};
	for (auto& programGroup : programGroups) {
		optixUtilAccumulateStackSizes(programGroup, &stackSizes, pipeline);
	}

	uint32_t directCallableStackSizeFromTraversal;
	uint32_t directCallableStackSizeFromState;
	uint32_t continuatuinStackSize;

	optixUtilComputeStackSizes(&stackSizes, maxTraceDepth, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuatuinStackSize);
	optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuatuinStackSize, 2);

	OptixShaderBindingTable shaderBindingTable = {};

	CUdeviceptr raygenRecord;
	const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
	cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize);
	RayGenSbtRecord raygenSBT;
	optixSbtRecordPackHeader(raygenProgramGroup, &raygenSBT);
	raygenSBT.data = { 1.0f, 0.0f, 0.0f };
	cudaMemcpy(reinterpret_cast<void*>(raygenRecord), &raygenSBT, raygenRecordSize, cudaMemcpyHostToDevice);

	CUdeviceptr missRecord;
	size_t missRecordSize = sizeof(MissSbtRecord);
	cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize);
	RayGenSbtRecord missSBT;
	optixSbtRecordPackHeader(missProgramGroup, &missSBT);
	cudaMemcpy(reinterpret_cast<void*>(missRecord), &missSBT, missRecordSize, cudaMemcpyHostToDevice);

	shaderBindingTable.raygenRecord = raygenRecord;
	shaderBindingTable.missRecordBase = missRecord;
	shaderBindingTable.missRecordStrideInBytes = sizeof(MissSbtRecord);
	shaderBindingTable.missRecordCount = 1;

	CudaOutputBuffer<uchar4> outBuffer = CudaOutputBuffer<uchar4>(width, height);

	CUstream stream;
	cudaStreamCreate(&stream);

	Params params;
	params.image = outBuffer.map();
	params.image_width = width;

	CUdeviceptr deviceParams;
	cudaMalloc(reinterpret_cast<void**>(&deviceParams), sizeof(Params));
	cudaMemcpy(reinterpret_cast<void*>(deviceParams), &params, sizeof(params), cudaMemcpyHostToDevice);

	optixLaunch(pipeline, stream, deviceParams, sizeof(Params), &shaderBindingTable, width, height, 1);
	cudaDeviceSynchronize();

	outBuffer.unmap();
	cudaFree(reinterpret_cast<void*>(deviceParams));

	//displayImage

}
*/