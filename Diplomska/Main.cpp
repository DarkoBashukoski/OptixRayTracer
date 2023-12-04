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
#include <array>

#include "DisplayManager.h"
#include "Renderer.h"
#include "Camera.h"
#include "Timer.h"

using namespace std;

void CUDA_CHECK(cudaError_t error) {
	if (error != cudaError::cudaSuccess) {
		cout << "Cuda error: " << error << endl;
	}
}

void printMatrix(mat4 mat) {
	cout << mat.m11 << ", " << mat.m12 << ", " << mat.m13 << ", " << mat.m14 << endl;
	cout << mat.m21 << ", " << mat.m22 << ", " << mat.m23 << ", " << mat.m24 << endl;
	cout << mat.m31 << ", " << mat.m32 << ", " << mat.m33 << ", " << mat.m34 << endl;
	cout << mat.m41 << ", " << mat.m42 << ", " << mat.m43 << ", " << mat.m44 << endl;
}

void OPTIX_CHECK(OptixResult error) {
	if (error != OptixResult::OPTIX_SUCCESS) {
		cout << "Optix error: " << error << endl;
	}
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
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

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

int main() {
	int width = 1920;
	int height = 1080;

	DisplayManager displayManager = DisplayManager();
	Renderer renderer = Renderer();

	OptixDeviceContext context = nullptr;
	CUDA_CHECK(cudaFree(0));
	CUcontext cuCtx = 0;
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	OptixTraversableHandle gasHandle;
	CUdeviceptr deviceGasOutputBuffer;
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
	accel_options.motionOptions.numKeys = 0; //No motion blur
	/*
	const array<float3, 36> vertices = { {
		  {  0.0f,  1.0f,  1.0f }, //predna strana
		  {  0.0f,  0.0f,  1.0f }, //predna strana
		  {  1.0f,  0.0f,  1.0f }, //predna strana
		  {  0.0f,  1.0f,  1.0f }, //predna strana
		  {  1.0f,  0.0f,  1.0f }, //predna strana
		  {  1.0f,  1.0f,  1.0f }, //predna strana

		  {  1.0f,  1.0f,  1.0f }, //desna strana
		  {  1.0f,  0.0f,  1.0f }, //desna strana 
		  {  1.0f,  0.0f,  0.0f }, //desna strana
		  {  1.0f,  1.0f,  1.0f }, //desna strana
		  {  1.0f,  0.0f,  0.0f }, //desna strana
		  {  1.0f,  1.0f,  0.0f }, //desna strana

		  {  0.0f,  1.0f,  0.0f }, //leva strana
		  {  0.0f,  0.0f,  0.0f }, //leva strana
		  {  0.0f,  0.0f,  1.0f }, //leva strana
		  {  0.0f,  1.0f,  0.0f }, //leva strana
		  {  0.0f,  0.0f,  1.0f }, //leva strana
		  {  0.0f,  1.0f,  1.0f }, //leva strana

		  {  1.0f,  1.0f,  0.0f }, //zadna strana
		  {  1.0f,  0.0f,  0.0f }, //zadna strana
		  {  0.0f,  0.0f,  0.0f }, //zadna strana
		  {  1.0f,  1.0f,  0.0f }, //zadna strana
		  {  0.0f,  0.0f,  0.0f }, //zadna strana
		  {  0.0f,  1.0f,  0.0f }, //zadna strana

		  {  0.0f,  0.0f,  1.0f }, //dolna strana
		  {  0.0f,  0.0f,  0.0f }, //dolna strana
		  {  1.0f,  0.0f,  0.0f }, //dolna strana
		  {  0.0f,  0.0f,  1.0f }, //dolna strana
		  {  1.0f,  0.0f,  0.0f }, //dolna strana
		  {  1.0f,  0.0f,  1.0f }, //dolna strana

		  {  0.0f,  1.0f,  0.0f }, //gorna strana
		  {  0.0f,  1.0f,  1.0f }, //gorna strana
		  {  1.0f,  1.0f,  1.0f }, //gorna strana
		  {  0.0f,  1.0f,  0.0f }, //gorna strana
		  {  1.0f,  1.0f,  1.0f }, //gorna strana
		  {  1.0f,  1.0f,  0.0f }, //gorna strana
	} };
	*/
	const array<float3, 8> vertices = { {
		{0.0f, 0.0f, 0.0f}, //0
		{1.0f, 0.0f, 0.0f}, //1
		{0.0f, 1.0f, 0.0f}, //2
		{0.0f, 0.0f, 1.0f}, //3
		{1.0f, 1.0f, 0.0f}, //4
		{1.0f, 0.0f, 1.0f}, //5
		{0.0f, 1.0f, 1.0f}, //6
		{1.0f, 1.0f, 1.0f}  //7
	} };

	const array<uint3, 12> indexBuffer = { {
		{6, 3, 5}, {6, 5, 7}, //predna strana
		{7, 5, 1}, {7, 1, 4}, //desna strana
		{2, 0, 3}, {2, 3, 6}, //leva strana
		{4, 1, 0}, {4, 0, 2}, //zadna strana
		{3, 0, 1}, {3, 1, 5}, //dolna strana
		{2, 6, 7}, {2, 7, 4}  //gorna strana
	} };

	const std::array<float3, 3> g_diffuse_colors = { {
		{ 1.0f, 0.0f, 0.0f }, //red   -- index 0
		{ 0.0f, 1.0f, 0.0f }, //green -- index 1
		{ 0.0f, 0.0f, 1.0f }, //blue  -- index 2
	} };

	static std::array<uint32_t, 12> g_mat_indices = { {
		0, 0,                          // predna        -- red lambert
		1, 1,                          // desna         -- green lambert
		1, 1,                          // leva wall     -- green lambert
		0, 0,                          // zadna wall    -- red lambert
		2, 2,                          // dolna wall    -- blue lambert
		2, 2,                          // gorna block   -- blue lambert
	} };

	const size_t verticesSize = sizeof(float3) * vertices.size();
	CUdeviceptr dVertices = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertices), verticesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dVertices), vertices.data(), verticesSize, cudaMemcpyHostToDevice));

	const size_t matSize = sizeof(uint32_t) * g_mat_indices.size();
	CUdeviceptr dMaterialIndeces = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMaterialIndeces), matSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMaterialIndeces), g_mat_indices.data(), matSize, cudaMemcpyHostToDevice));

	const size_t indicesSize = sizeof(uint3) * indexBuffer.size();
	CUdeviceptr dIndexBuffer = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIndexBuffer), indicesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dIndexBuffer), indexBuffer.data(), indicesSize, cudaMemcpyHostToDevice));

	//one flag per sbt record
	const uint32_t triangle_input_flags[3] = { OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE };

	OptixBuildInput triangle_input = {};
	triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
	triangle_input.triangleArray.vertexBuffers = &dVertices; //trebit da se smenit na novite -- vekje e na novite
	triangle_input.triangleArray.indexBuffer = dIndexBuffer;
	triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indexBuffer.size());
	triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
	triangle_input.triangleArray.flags = triangle_input_flags;
	triangle_input.triangleArray.numSbtRecords = 3;
	triangle_input.triangleArray.sbtIndexOffsetBuffer = dMaterialIndeces;
	triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	OptixAccelBufferSizes gasBufferSizes; //TODO compacting later
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gasBufferSizes));
	CUdeviceptr dTempBufferGas;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferGas), gasBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGasOutputBuffer), gasBufferSizes.outputSizeInBytes));

	OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, dTempBufferGas, gasBufferSizes.tempSizeInBytes, deviceGasOutputBuffer, gasBufferSizes.outputSizeInBytes, &gasHandle, nullptr, 0));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTempBufferGas)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dVertices)));




	//IAS BUILD

	float identityMatrix[24] = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f, 2.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f
	};

	std::vector<OptixInstance> instances;

	for (int i = 0; i < 2; i++) {
		OptixInstance optixInstance = {};
		optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		memcpy(optixInstance.transform, &identityMatrix[i * 12], sizeof(float) * 12);
		optixInstance.visibilityMask = 255;
		optixInstance.sbtOffset = i;
		optixInstance.instanceId = 0;
		optixInstance.traversableHandle = gasHandle;
		instances.push_back(optixInstance);
	}

	CUdeviceptr d_optixInstance;
	const size_t optixInstancesSize = sizeof(OptixInstance) * instances.size();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_optixInstance), optixInstancesSize));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_optixInstance),
		&instances[0],
		optixInstancesSize,
		cudaMemcpyHostToDevice
	));

	OptixBuildInput instanceInput = {};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = d_optixInstance;
	instanceInput.instanceArray.numInstances = instances.size();

	OptixAccelBuildOptions iasAccelOptions = {};
	iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes ias_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &iasAccelOptions, &instanceInput, 1, &ias_buffer_sizes));
	CUdeviceptr d_temp_buffer_ias;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_ias), ias_buffer_sizes.tempSizeInBytes));

	CUdeviceptr d_ias_output_buffer = 0;
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_output_buffer)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ias_output_buffer), ias_buffer_sizes.outputSizeInBytes));

	OptixTraversableHandle iasHandle;

	OPTIX_CHECK(optixAccelBuild(
		context,
		0,
		&iasAccelOptions,
		&instanceInput,
		1,
		d_temp_buffer_ias,
		ias_buffer_sizes.tempSizeInBytes,
		d_ias_output_buffer,
		ias_buffer_sizes.outputSizeInBytes,
		&iasHandle,
		nullptr,
		0
	));

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_ias)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_optixInstance)));

	//IAS BUILD




	OptixModule module = nullptr;
	OptixPipelineCompileOptions pipeline_compile_options = {};
	OptixModuleCompileOptions module_compile_options = {};

	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeline_compile_options.numPayloadValues = 3; // memory size of payload (in trace())
	pipeline_compile_options.numAttributeValues = 3; // memory size of attributes (from is())
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	//pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	string location;
	string cudaSourceCode;
	const char* fileName = "DrawSolidColor.cu";

	ifstream file = ifstream(fileName, ios::binary);
	vector<unsigned char> buffer = vector<unsigned char>(istreambuf_iterator<char>(file), {});
	cudaSourceCode.assign(buffer.begin(), buffer.end());

	nvrtcProgram prog = 0;

	nvrtcCreateProgram(&prog, cudaSourceCode.c_str(), fileName, 0, NULL, NULL);

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

	nvrtcResult result = nvrtcCompileProgram(prog, includeOptions.size(), includeOptions.data());

	
	size_t errorLogForNVRTCSize;
	nvrtcGetProgramLogSize(prog, &errorLogForNVRTCSize);
	char* nvrtcErrorlog = (char*)malloc(sizeof(char) * errorLogForNVRTCSize);
	nvrtcGetProgramLog(prog, nvrtcErrorlog);
	cout << nvrtcErrorlog << endl;
	

	size_t compiledCodeSize;
	nvrtcGetOptiXIRSize(prog, &compiledCodeSize);
	char* compiledCode = new char[compiledCodeSize];
	nvrtcGetOptiXIR(prog, compiledCode);
	nvrtcDestroyProgram(&prog);

	char* outLog = (char*)malloc(sizeof(char) * 2048);
	size_t outLogSize = 0;

	OPTIX_CHECK(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, compiledCode, compiledCodeSize, outLog, &outLogSize, &module));
	
	OptixProgramGroup raygen_prog_group = nullptr;
	OptixProgramGroup miss_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_group_red = nullptr;
	OptixProgramGroup hitgroup_prog_group_green = nullptr;
	OptixProgramGroup hitgroup_prog_group_blue = nullptr;

	OptixProgramGroupOptions program_group_options = {};

	OptixProgramGroupDesc raygen_prog_group_desc = {};
	raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygen_prog_group_desc.raygen.module = module;
	raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
	OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_prog_group_desc, 1, &program_group_options, outLog, &outLogSize, &raygen_prog_group));
	
	OptixProgramGroupDesc miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = module;
	miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
	OPTIX_CHECK(optixProgramGroupCreate(context, &miss_prog_group_desc, 1, &program_group_options, outLog, &outLogSize, &miss_prog_group));

	OptixProgramGroupDesc hitgroup_prog_group_desc = {};
	hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_prog_group_desc.hitgroup.moduleCH = module;
	hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc, 1, &program_group_options, outLog, &outLogSize, &hitgroup_prog_group));

	OptixProgramGroupDesc hitgroup_prog_group_desc_red = {};
	hitgroup_prog_group_desc_red.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_prog_group_desc_red.hitgroup.moduleCH = module;
	hitgroup_prog_group_desc_red.hitgroup.entryFunctionNameCH = "__closesthit__red";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc_red, 1, &program_group_options, outLog, &outLogSize, &hitgroup_prog_group_red));

	OptixProgramGroupDesc hitgroup_prog_group_desc_green = {};
	hitgroup_prog_group_desc_green.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_prog_group_desc_green.hitgroup.moduleCH = module;
	hitgroup_prog_group_desc_green.hitgroup.entryFunctionNameCH = "__closesthit__green";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc_green, 1, &program_group_options, outLog, &outLogSize, &hitgroup_prog_group_green));

	OptixProgramGroupDesc hitgroup_prog_group_desc_blue = {};
	hitgroup_prog_group_desc_blue.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroup_prog_group_desc_blue.hitgroup.moduleCH = module;
	hitgroup_prog_group_desc_blue.hitgroup.entryFunctionNameCH = "__closesthit__blue";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc_blue, 1, &program_group_options, outLog, &outLogSize, &hitgroup_prog_group_blue));

	OptixPipeline pipeline = nullptr;
	const uint32_t    max_trace_depth = 1;
	OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group, hitgroup_prog_group_red, hitgroup_prog_group_green, hitgroup_prog_group_blue };
	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = max_trace_depth;
	OPTIX_CHECK(optixPipelineCreate(context, &pipeline_compile_options,	&pipeline_link_options,	program_groups,	sizeof(program_groups) / sizeof(program_groups[0]),	outLog, &outLogSize, &pipeline));

	OptixStackSizes stack_sizes = {};
	for (auto& prog_group : program_groups) {
		OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
	}

	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &direct_callable_stack_size_from_traversal,	&direct_callable_stack_size_from_state, &continuation_stack_size));
	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size, 2));

	OptixShaderBindingTable sbt = {};

	CUdeviceptr raygen_record;
	const size_t raygen_record_size = sizeof(RayGenSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
	RayGenSbtRecord rg_sbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

	CUdeviceptr miss_record;
	size_t miss_record_size = sizeof(MissSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
	MissSbtRecord ms_sbt;
	ms_sbt.data.topColor = { 0.3f, 0.1f, 0.2f };
	ms_sbt.data.bottomColor = { 0.5f, 0.6f, 0.7f };
	optixSbtRecordPackHeader(miss_prog_group, &ms_sbt);
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),	&ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

	const int hitgroupRecordCount = 4;
	CUdeviceptr hitgroup_record;
	size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size * hitgroupRecordCount));
	HitGroupSbtRecord hg_sbt[hitgroupRecordCount];

	OptixProgramGroup hitPrograms[3] = { hitgroup_prog_group_red , hitgroup_prog_group_green , hitgroup_prog_group_blue };

	for (int i = 0; i < 3; i++) {
		OPTIX_CHECK(optixSbtRecordPackHeader(hitPrograms[i], &hg_sbt[i]));
		hg_sbt[i].data.matColor = g_diffuse_colors[i];
	}

	OPTIX_CHECK(optixSbtRecordPackHeader(hitPrograms[2], &hg_sbt[3]));
	hg_sbt[3].data.matColor = g_diffuse_colors[2];

	CUDA_CHECK(cudaMemcpy( reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size * hitgroupRecordCount,	cudaMemcpyHostToDevice));

	sbt.raygenRecord = raygen_record;
	sbt.missRecordBase = miss_record;
	sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = hitgroup_record;
	sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	sbt.hitgroupRecordCount = hitgroupRecordCount;

	CudaOutputBuffer<uchar4> output_buffer(width, height);
	CudaOutputBuffer<uint3> debug_buffer(width, height);

	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	float3 camPos = make_float3(0.0f, 0.0f, 5.0f);
	float3 camRot = make_float3(0.0f, 0.0f, 0.0f);

	Camera camera = Camera(camPos, camRot, displayManager.getWindow());

	Params params;
	params.image = output_buffer.map();
	params.debug_buffer = debug_buffer.map();
	params.width = width;
	params.height = height;
	params.handle = iasHandle;

	unsigned int* d_rngState;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rngState), sizeof(unsigned int)));
	params.rngState = d_rngState;

	CUdeviceptr d_param;

	//ofstream myfile;
	//myfile.open("example.txt");
	
	while (!displayManager.isCloseRequested()) {		
		camera.update();
		params.camPosition = camera.getPosition();
		params.projectionMatrix = camera.getProjectionMatrix();
		params.viewMatrix = camera.getViewMatrix();
		params.inverseProjection = camera.getProjectionMatrix().inverse();
		params.inverseView = camera.getViewMatrix().inverse();

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));
		OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1));

		CUDA_CHECK(cudaDeviceSynchronize());
		output_buffer.unmap();
		debug_buffer.unmap();
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));

		//uchar4* hostPointer = output_buffer.getHostPointer();
		//uint3* debugPointer = debug_buffer.getHostPointer();

		//for (int i = 0; i < 1920 * 1080; i++) {
		//	myfile << (int)(*(hostPointer + i)).x << ", " << (int)(*(hostPointer + i)).y << ", " << (int)(*(hostPointer + i)).z << endl;
		//}

		renderer.render(output_buffer);
		displayManager.updateDisplay();
		Timer::getInstance()->update();
		cout << Timer::getInstance()->getFPS() << endl;

		//break;
	}

	displayManager.closeDisplay();
}