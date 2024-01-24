#pragma once
#ifndef OPTIXMANAGER_H
#define OPTIXMANAGER_H

#include <optix/optix.h>
#include <optix/optix_stack_size.h>
#include <optix/optix_stubs.h>

#include "RawModel.h"
#include "Entity.h"
#include "ErrorChecks.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <unordered_map>
#include "OptixShaderCompiler.h"
#include "Parameters.h"

template <typename T>
struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RaygenData>   RaygenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitgroupData> HitgroupSbtRecord;

class OptixManager {
private:
	static OptixManager* instance;
	OptixManager();
	OptixDeviceContext context = nullptr;
	OptixModule module = nullptr;
	OptixPipeline pipeline = nullptr;
	OptixShaderBindingTable sbt = {};
	char* outLog;
	size_t outLogSize = 0;
	const uint32_t maxTraceDepth = 5;
	OptixTraversableHandle iasHandle;
	unordered_map<RawModel*, vector<Entity>> entities;
	unsigned int totalTriangleCount;

	OptixProgramGroup raygenProgGroup = nullptr;
	OptixProgramGroup missProgGroup = nullptr;
	OptixProgramGroup hitgroupProgGroup = nullptr;
	OptixProgramGroup hitgroupDielectricProgGroup = nullptr;

	void buildModule();
	void buildPipeline(OptixPipelineCompileOptions pipelineCompileOptions);
	uint32_t getHitgroupRecordCount();
public:
	OptixManager(const OptixManager& obj) = delete;
	static OptixManager* getInstance();
	OptixDeviceContext& getContext();
	OptixTraversableHandle getIasHandle();
	OptixShaderBindingTable getSbt();
	OptixPipeline getPipeline();
	void addEntity(Entity entity);
	void addEntities(vector<Entity> _entities);
	void buildIas();
	void buildSbt();
	unsigned int getTriangleCount();
};

#endif // !OPTIXMANAGER_H

