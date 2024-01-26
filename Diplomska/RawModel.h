#pragma once
#ifndef RAWMODEL_H
#define RAWMODEL_H

#include <vector_types.h>
#include <optix/optix_types.h>
#include <string>
#include <fstream>
#include <sstream>
#include "json.hpp"
#include <iostream>
#include "ErrorChecks.h"
#include <cuda_runtime.h>
#include "Material.h"
#include <optix/optix_stubs.h>

using namespace std;
using json = nlohmann::json;

class RawModel {
private:
	OptixDeviceContext context;

	uint32_t vertexCount;
	float3* vertices;
	uint32_t vertexNormalCount;
	float3* vertexNormals;
	uint32_t triangleCount;
	uint3* indices;
	uint3* vertexNormalIndices;
	OptixTraversableHandle gasHandle;
	Material material;

	CUdeviceptr dVertexNormals;
	CUdeviceptr dVertexNormalIndices;

	void parseModelData(json modelData);
	void parseMaterialData(json materialData);
	void buildGas();
public:
	RawModel(OptixDeviceContext _context, string fileName);
	OptixTraversableHandle getGasHandle();
	Material& getMaterial();
	CUdeviceptr getDeviceVertexNormals();
	CUdeviceptr getDeviceVertexNormalIndices();
	uint32_t getTriangleCount();
	float3* getVertices();
	float3* getNormals();
	uint3* getIndices();
};

#endif // !RAWMODEL_H
