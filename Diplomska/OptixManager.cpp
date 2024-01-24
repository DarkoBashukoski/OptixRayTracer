#include "OptixManager.h"

OptixManager* OptixManager::instance = nullptr;

OptixManager::OptixManager() {
	CUDA_CHECK(cudaFree(0));
	CUcontext cuCtx = 0;
	OPTIX_CHECK(optixInit());

	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;

	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	outLog = (char*)malloc(sizeof(char) * 2048);

	buildModule();
}

void OptixManager::buildModule() {
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixModuleCompileOptions moduleCompileOptions = {};

	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.numPayloadValues = 15;
	pipelineCompileOptions.numAttributeValues = 3;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	size_t compiledCodeSize;
	char* compiledCode = OptixShaderCompiler::compileShader("RayTracing.cu", &compiledCodeSize);

	OPTIX_CHECK(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, compiledCode, compiledCodeSize, outLog, &outLogSize, &module));
	buildPipeline(pipelineCompileOptions);
}

void OptixManager::buildPipeline(OptixPipelineCompileOptions pipelineCompileOptions) {
	OptixProgramGroupOptions programGroupOptions = {};

	OptixProgramGroupDesc raygenProgGroupDesc = {};
	raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenProgGroupDesc.raygen.module = module;
	raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__rg";
	OPTIX_CHECK(optixProgramGroupCreate(context, &raygenProgGroupDesc, 1, &programGroupOptions, outLog, &outLogSize, &raygenProgGroup));

	OptixProgramGroupDesc missProgGroupDesc = {};
	missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	missProgGroupDesc.miss.module = module;
	missProgGroupDesc.miss.entryFunctionName = "__miss__ms";
	OPTIX_CHECK(optixProgramGroupCreate(context, &missProgGroupDesc, 1, &programGroupOptions, outLog, &outLogSize, &missProgGroup));

	OptixProgramGroupDesc hitgroupProgGroupDesc = {};
	hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroupProgGroupDesc.hitgroup.moduleCH = module;
	hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroupProgGroupDesc, 1, &programGroupOptions, outLog, &outLogSize, &hitgroupProgGroup));

	OptixProgramGroupDesc hitgroupDielectricProgGroupDesc = {};
	hitgroupDielectricProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hitgroupDielectricProgGroupDesc.hitgroup.moduleCH = module;
	hitgroupDielectricProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__dielectric";
	OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroupDielectricProgGroupDesc, 1, &programGroupOptions, outLog, &outLogSize, &hitgroupDielectricProgGroup));

	OptixProgramGroup programGroups[] = { raygenProgGroup, missProgGroup, hitgroupProgGroup, hitgroupDielectricProgGroup };
	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
	OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, sizeof(programGroups) / sizeof(programGroups[0]), outLog, &outLogSize, &pipeline));

	OptixStackSizes stackSizes = {};
	for (OptixProgramGroup progGroup : programGroups) {
		OPTIX_CHECK(optixUtilAccumulateStackSizes(progGroup, &stackSizes, pipeline));
	}

	uint32_t directCallableStackSizeFromTraversal;
	uint32_t directCallableStackSizeFromState;
	uint32_t continuationStackSize;
	OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, maxTraceDepth, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuationStackSize));
	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 2));
}

OptixManager* OptixManager::getInstance() {
	if (instance != nullptr) {
		return instance;
	}
	instance = new OptixManager();
	return instance;
}

OptixDeviceContext& OptixManager::getContext() {
	return context;
}

OptixTraversableHandle OptixManager::getIasHandle() {
	return iasHandle;
}

OptixShaderBindingTable OptixManager::getSbt() {
	return sbt;
}

OptixPipeline OptixManager::getPipeline() {
	return pipeline;
}

void OptixManager::addEntity(Entity entity) {
	RawModel* model = entity.getModel();
	if (entities.find(model) == entities.end()) {
		entities.insert({ model, vector<Entity>() });
	}
	entities[model].push_back(entity);
	totalTriangleCount += model->getTriangleCount();
}

void OptixManager::addEntities(vector<Entity> _entities) {
	for (Entity e : _entities) {
		addEntity(e);
	}
}

unsigned int OptixManager::getTriangleCount() {
	return totalTriangleCount;
}

void OptixManager::buildIas() {
	std::vector<OptixInstance> instances;
	int sbtOffset = 0;

	for (const pair<RawModel*, vector<Entity>> pair : entities) {
		RawModel* model = pair.first;
		for (Entity entity : pair.second) {
			OptixInstance optixInstance = {};
			optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
			memcpy(optixInstance.transform, entity.getTransformation(), sizeof(float) * 12);
			optixInstance.visibilityMask = 255;
			optixInstance.sbtOffset = sbtOffset;
			optixInstance.instanceId = 0;
			optixInstance.traversableHandle = model->getGasHandle();

			instances.push_back(optixInstance);
		}
		sbtOffset += 1;
	}

	CUdeviceptr dOptixInstances;
	size_t optixInstancesSize = sizeof(OptixInstance) * instances.size();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOptixInstances), optixInstancesSize));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dOptixInstances), instances.data(), optixInstancesSize, cudaMemcpyHostToDevice));

	OptixBuildInput instanceInput = {};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = dOptixInstances;
	instanceInput.instanceArray.numInstances = instances.size();

	OptixAccelBuildOptions iasAccelOptions = {};
	iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE; //smeni na compression pokasno
	iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes iasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &iasAccelOptions, &instanceInput, 1, &iasBufferSizes));

	CUdeviceptr dTempBufferIas;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferIas), iasBufferSizes.tempSizeInBytes));

	CUdeviceptr dIasOutputBuffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIasOutputBuffer), iasBufferSizes.outputSizeInBytes));  //trebit free nekade probably

	OPTIX_CHECK(optixAccelBuild(context, 0, &iasAccelOptions, &instanceInput, 1, dTempBufferIas, iasBufferSizes.tempSizeInBytes, dIasOutputBuffer, iasBufferSizes.outputSizeInBytes, &iasHandle, nullptr, 0));

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dOptixInstances)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTempBufferIas)));
}

void OptixManager::buildSbt() {
	CUdeviceptr dRaygenRecord;
	size_t raygenRecordSize = sizeof(RaygenSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRaygenRecord), raygenRecordSize)); //trebit free nekade
	RaygenSbtRecord raygenRecord;
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgGroup, &raygenRecord));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dRaygenRecord), &raygenRecord, raygenRecordSize, cudaMemcpyHostToDevice));

	CUdeviceptr dMissRecord;
	size_t missRecordSize = sizeof(MissSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMissRecord), missRecordSize));
	MissSbtRecord missRecord;
	missRecord.data.skyColorHorizon = { 1.0f, 1.0f, 1.0f };
	missRecord.data.skyColorZenith = { 0.309f, 0.18f, 0.874f };
	missRecord.data.groundColor = { 0.65f, 0.882f, 0.337 };
	missRecord.data.sunDirection = { 0.0f, 0.0f, -1.0f };
	missRecord.data.sunFocus = 500;
	missRecord.data.sunIntensity = 50;
	optixSbtRecordPackHeader(missProgGroup, &missRecord);
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMissRecord), &missRecord, missRecordSize, cudaMemcpyHostToDevice));

	CUdeviceptr dHitgroupRecord;
	size_t hitgroupRecordSize = sizeof(HitgroupSbtRecord);
	vector<HitgroupSbtRecord> hitgroupRecords = vector<HitgroupSbtRecord>();

	for (const pair<RawModel*, vector<Entity>> pair : entities) {
		RawModel* model = pair.first;
		Material material = model->getMaterial();

		HitgroupSbtRecord newRecord = {};

		OPTIX_CHECK(optixSbtRecordPackHeader((material.shaderId == 0) ? hitgroupProgGroup : hitgroupDielectricProgGroup, &newRecord));
		newRecord.data.color = material.color;
		newRecord.data.metallic = material.metallic;
		newRecord.data.roughness = material.roughness;
		newRecord.data.emissionColor = material.emissionColor;
		newRecord.data.emissionPower = material.emissionPower;

		newRecord.data.vertexNormals = (float3*)model->getDeviceVertexNormals();
		newRecord.data.vertexNormalIndices = (uint3*)model->getDeviceVertexNormalIndices();

		hitgroupRecords.push_back(newRecord);
	}
	
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dHitgroupRecord), hitgroupRecordSize * hitgroupRecords.size()));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dHitgroupRecord), hitgroupRecords.data(), hitgroupRecordSize * hitgroupRecords.size(), cudaMemcpyHostToDevice));

	sbt.raygenRecord = dRaygenRecord;
	sbt.missRecordBase = dMissRecord;
	sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = dHitgroupRecord;
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupSbtRecord);
	sbt.hitgroupRecordCount = hitgroupRecords.size();
}

uint32_t OptixManager::getHitgroupRecordCount() {
	uint32_t total = 0;

	for (const pair<RawModel*, vector<Entity>> pair : entities) {
		RawModel* model = pair.first;
		
		total += 1;
	}

	return total;
}
