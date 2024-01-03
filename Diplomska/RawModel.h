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
	uint32_t triangleCount;
	uint3* indices;
	uint32_t* materialIndices;
	OptixTraversableHandle gasHandle;
	uint32_t materialCount;
	vector<Material> materials;

	void parseModelData(json modelData);
	void parseMaterialData(json materialData);
	void buildGas();
public:
	RawModel(OptixDeviceContext _context, string fileName);
	OptixTraversableHandle getGasHandle();
	uint32_t getMaterialCount();
	vector<Material> getMaterials();
};

#endif // !RAWMODEL_H
